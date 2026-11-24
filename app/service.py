"""格网登记与雨量换算的服务层。

设计要点
--------
* 登记号在本次运行内有效；每个登记号保存登记当时的格网几何快照，
  之后即便用“同名含义”的新参数再次登记，旧号仍按原几何使用。
* 交叠表以源/目标几何元组为缓存键：原点、尺寸或行列数任意一项变化，
  键即不同，会重新计算并生成新登记号；完全相同的几何才复用交叠关系。
* 缺测（null）范围不做缓存，每次换算按当天输入逐格判断。
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass
from fractions import Fraction

from .geometry import (
    GridMetrics,
    OverlapPair,
    OverlapTable,
    as_decimal_fraction,
    build_overlap,
)
from .schemas import (
    ConservationInfo,
    GridSpec,
    RegisterResponse,
    TargetCellResult,
    TransferResponse,
)

_MM_TO_M = Fraction(1, 1000)  # 雨深毫米 → 米


class RegistrationNotFoundError(KeyError):
    """登记号不存在。"""


@dataclass(frozen=True)
class Registration:
    registration_id: str
    source: GridMetrics
    target: GridMetrics
    overlap: OverlapTable


class Registry:
    """线程安全的运行内登记处。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, Registration] = {}
        # 相同几何（源, 目标）→ 已有的登记号，用于复用交叠关系。
        self._by_geometry: dict[tuple[GridMetrics, GridMetrics], str] = {}

    def register(self, source_spec: GridSpec, target_spec: GridSpec) -> RegisterResponse:
        source = source_spec.to_metrics()
        target = target_spec.to_metrics()
        key = (source, target)

        with self._lock:
            existing_id = self._by_geometry.get(key)
            if existing_id is not None:
                registration = self._by_id[existing_id]
                return RegisterResponse(
                    registration_id=existing_id,
                    overlap_pair_count=len(registration.overlap.pairs),
                    overlap_area_m2=float(registration.overlap.total_overlap_area),
                    reused=True,
                )

            overlap = build_overlap(source, target)
            registration_id = uuid.uuid4().hex
            registration = Registration(
                registration_id=registration_id,
                source=source,
                target=target,
                overlap=overlap,
            )
            self._by_id[registration_id] = registration
            self._by_geometry[key] = registration_id

        return RegisterResponse(
            registration_id=registration_id,
            overlap_pair_count=len(overlap.pairs),
            overlap_area_m2=float(overlap.total_overlap_area),
            reused=False,
        )

    def get(self, registration_id: str) -> Registration:
        registration = self._by_id.get(registration_id)
        if registration is None:
            raise RegistrationNotFoundError(registration_id)
        return registration

    def count(self) -> int:
        return len(self._by_id)

    def transfer(
        self,
        registration_id: str,
        depth_mm: list[list[float | None]],
    ) -> TransferResponse:
        registration = self.get(registration_id)
        source = registration.source
        target = registration.target
        overlap = registration.overlap

        if len(depth_mm) != source.rows or any(
            len(row) != source.cols for row in depth_mm
        ):
            raise ValueError(
                f"雨深矩阵维度应为 {source.rows} 行 × {source.cols} 列，"
                f"实际为 {len(depth_mm)} 行 × "
                f"{len(depth_mm[0]) if depth_mm else 0} 列"
            )

        missing = sum(1 for row in depth_mm for value in row if value is None)

        # 雨深按十进制值转成精确有理数；再统一换成同一整数刻度上的整数，
        # 逐对汇总时只做整数乘加，最后一次性还原为有理数体积，避免
        # 逐对 Fraction 约分带来的误差与开销。
        exact_depth: list[list[Fraction | None]] = [
            [None if value is None else as_decimal_fraction(value) for value in row]
            for row in depth_mm
        ]
        depth_denominator = 1
        for row in exact_depth:
            for depth in row:
                if depth is not None:
                    depth_denominator = math.lcm(depth_denominator, depth.denominator)
        depth_units: list[list[int | None]] = [
            [
                None if depth is None else int(depth * depth_denominator)
                for depth in row
            ]
            for row in exact_depth
        ]

        denominator = overlap.area_denominator
        source_cell_area = overlap.source_cell_area_units
        target_cell_area = overlap.target_cell_area_units
        # 体积（m³）的公共分母：面积刻度 × 雨深刻度 × 毫米→米。
        volume_denominator = denominator * depth_denominator * 1000

        # 每个目标格按源格累计有效面积（整数刻度）与有效水量（整数刻度）。
        effective_area_units: dict[tuple[int, int], int] = {}
        effective_volume_units: dict[tuple[int, int], int] = {}

        # 每个源格已有多少面积单位被目标范围接收（仅统计有记录且实际相交的部分）。
        transferred_area_by_source: dict[tuple[int, int], int] = {}

        # 有记录源格的总水量（按源格完整面积核算，便于守恒核对）。
        recorded_source_volume_units = 0
        for row in depth_units:
            for depth in row:
                if depth is not None:
                    recorded_source_volume_units += source_cell_area * depth

        # 交叠表仅含实际相交格对；缺测源格当天直接跳过，
        # 其交叠面积不计入任何目标格的有效面积。
        transferred_volume_units = 0
        pair: OverlapPair
        for pair in overlap.pairs:
            depth = depth_units[pair.source_row][pair.source_col]
            if depth is None:
                continue
            cell_key = (pair.target_row, pair.target_col)
            effective_area_units[cell_key] = (
                effective_area_units.get(cell_key, 0) + pair.area_units
            )
            volume_units = pair.area_units * depth
            effective_volume_units[cell_key] = (
                effective_volume_units.get(cell_key, 0) + volume_units
            )
            transferred_volume_units += volume_units

            source_key = (pair.source_row, pair.source_col)
            transferred_area_by_source[source_key] = (
                transferred_area_by_source.get(source_key, 0) + pair.area_units
            )

        # 目标外水量按“源格落在目标外的面积 × 当天雨深”逐格核算，
        # 而不是用总量相减，避免任何窄条上的雨量被错算或抹掉。
        outside_volume_units = 0
        for sr, row in enumerate(depth_units):
            for sc, depth in enumerate(row):
                if depth is None:
                    continue
                outside_area_units = source_cell_area - transferred_area_by_source.get(
                    (sr, sc), 0
                )
                if outside_area_units > 0:
                    outside_volume_units += outside_area_units * depth

        results: list[TargetCellResult] = []
        # 排序保证输出稳定、可对照矩阵位置阅读。
        for (tr, tc) in sorted(effective_area_units):
            area_units = effective_area_units[(tr, tc)]
            if area_units <= 0:
                continue
            volume_units = effective_volume_units[(tr, tc)]
            volume = Fraction(volume_units, volume_denominator)
            area = Fraction(area_units, denominator)
            coverage = Fraction(area_units, target_cell_area)
            # 精确算术下单个目标格的有效覆盖面积不会超过其自身面积；
            # 这里只防御极端上溢。
            if coverage > 1:
                coverage = Fraction(1)
            results.append(
                TargetCellResult(
                    target_row=tr,
                    target_col=tc,
                    effective_volume_m3=float(volume),
                    mean_depth_mm=float(volume / (area * _MM_TO_M)),
                    effective_coverage_ratio=float(coverage),
                    effective_area_m2=float(area),
                )
            )

        # 各分量均为整数刻度上的整数，无舍入；
        # 因此残差在精确算术下恒为 0。
        residual_units = recorded_source_volume_units - (
            transferred_volume_units + outside_volume_units
        )

        return TransferResponse(
            registration_id=registration_id,
            source_rows=source.rows,
            source_cols=source.cols,
            target_rows=target.rows,
            target_cols=target.cols,
            target_cell_area_m2=float(Fraction(target_cell_area, denominator)),
            missing_source_cells=missing,
            results=results,
            conservation=ConservationInfo(
                recorded_source_volume_m3=float(
                    Fraction(recorded_source_volume_units, volume_denominator)
                ),
                transferred_volume_m3=float(
                    Fraction(transferred_volume_units, volume_denominator)
                ),
                recorded_volume_outside_target_m3=float(
                    Fraction(outside_volume_units, volume_denominator)
                ),
                conservation_residual_m3=float(
                    Fraction(residual_units, volume_denominator)
                ),
            ),
        )
