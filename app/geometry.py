"""轴对齐矩形格网的交叠几何计算。

两张格网处于同一个平面米制坐标系中，均为轴对齐的规则矩形格网。
本模块只做纯几何计算，不涉及雨量与 HTTP 语义。

约定：格网原点 (origin_x, origin_y) 是第 (0, 0) 格的最小角坐标，
行沿 y 正方向递增，列沿 x 正方向递增。第 (r, c) 格覆盖：

    [origin_x + c*cell_width,  origin_x + (c+1)*cell_width]
    [origin_y + r*cell_height, origin_y + (r+1)*cell_height]

小数坐标与格宽高按**十进制米值**理解（例如 0.3 表示 3/10 米），
内部先把它们换成整数刻度上的精确整数/有理数，最后才在 API 边界
转回浮点数。这样既不会出现“完全对齐却因覆盖比例
1.0000000000000002 被拒”，也不会让浮点舍入误差把少量水量错记到
目标范围之外。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from fractions import Fraction
from typing import NamedTuple

# 浮点输入按“十进制米值”理解时保留的有效数字位数。
# 15 位足以吸收双精度加法的典型尾差（0.1 + 0.2 → 0.30000000000000004），
# 又远高于实际米制坐标/格宽的书写精度，不会吞掉真实差异。
_DECIMAL_PRECISION = 15


def as_decimal_fraction(value: int | float | str | Decimal | Fraction) -> Fraction:
    """把十进制米值转成精确有理数。

    浮点数先按其最短十进制表示（``str``）理解，再规范到
    ``_DECIMAL_PRECISION`` 位有效数字；因此 ``0.3`` 会变成 ``3/10``，
    而 ``0.1 + 0.2`` 的二进制尾差也会归回 ``3/10``，与用户书写的
    十进制米值一致。
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):  # bool 是 int 的子类，但不应作为长度/坐标。
        raise TypeError("布尔值不能作为米制数值")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("米制数值必须是有限值")
        decimal_value = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("米制数值必须是有限值")
        decimal_value = Decimal(str(value))
    elif isinstance(value, str):
        decimal_value = Decimal(value)
        if not decimal_value.is_finite():
            raise ValueError("米制数值必须是有限值")
    else:
        raise TypeError(f"不支持的米制数值类型: {type(value).__name__}")

    with localcontext() as ctx:
        ctx.prec = _DECIMAL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN
        return Fraction(+decimal_value)


def canonical_decimal_float(
    value: int | float | str | Decimal | Fraction,
) -> float:
    """返回十进制米值对应的规范浮点数。

    例如 ``0.1 + 0.2`` 会规范成 ``0.3``。登记的几何快照与缓存键都用
    规范值，保证“同一组十进制米值”无论经历怎样的浮点运算尾差，
    都得到同一份几何并复用交叠关系。
    """

    return float(as_decimal_fraction(value))


class GridMetrics(NamedTuple):
    """不可变的格网几何定义（坐标单位：米）。

    构造参数保持浮点/整数形态以兼容外部调用；需要精确计算时使用
    ``exact_*`` 属性，它们把十进制米值转换成分数。
    """

    origin_x: float
    origin_y: float
    cell_width: float
    cell_height: float
    cols: int
    rows: int

    @property
    def exact_origin_x(self) -> Fraction:
        return as_decimal_fraction(self.origin_x)

    @property
    def exact_origin_y(self) -> Fraction:
        return as_decimal_fraction(self.origin_y)

    @property
    def exact_cell_width(self) -> Fraction:
        return as_decimal_fraction(self.cell_width)

    @property
    def exact_cell_height(self) -> Fraction:
        return as_decimal_fraction(self.cell_height)

    @property
    def max_x(self) -> float:
        return self.origin_x + self.cols * self.cell_width

    @property
    def max_y(self) -> float:
        return self.origin_y + self.rows * self.cell_height

    @property
    def exact_max_x(self) -> Fraction:
        return self.exact_origin_x + self.cols * self.exact_cell_width

    @property
    def exact_max_y(self) -> Fraction:
        return self.exact_origin_y + self.rows * self.exact_cell_height

    @property
    def cell_area(self) -> float:
        return self.cell_width * self.cell_height

    @property
    def exact_cell_area(self) -> Fraction:
        return self.exact_cell_width * self.exact_cell_height


class OverlapPair(NamedTuple):
    """一个实际相交的（目标格, 源格）对及其交叠面积。

    面积以整数刻度保存：``area_units / area_denominator`` 平方米，
    避免在逐对汇总时反复做 Fraction 约分。``area_m2`` 以精确有理数暴露。
    """

    target_row: int
    target_col: int
    source_row: int
    source_col: int
    area_units: int
    area_denominator: int

    @property
    def area_m2(self) -> Fraction:
        return Fraction(self.area_units, self.area_denominator)


@dataclass(frozen=True)
class OverlapTable:
    """两张格网之间全部实际相交格对的不可变交叠表。"""

    source: GridMetrics
    target: GridMetrics
    pairs: tuple[OverlapPair, ...]
    # 所有格对面积共用的分母（平方米），便于整数累加。
    area_denominator: int = 1
    # 单个源格/目标格的面积，同样以 area_denominator 为分母。
    source_cell_area_units: int = 0
    target_cell_area_units: int = 0

    @property
    def total_overlap_area(self) -> Fraction:
        """所有格对交叠面积之和，即两张格网整体的交叠面积。"""
        return Fraction(
            sum(pair.area_units for pair in self.pairs), self.area_denominator
        )


def _axis_denominator(origin: Fraction, cell_size: Fraction) -> int:
    """该轴上所有格线都能用 1/D 表示的公共分母 D。"""
    return math.lcm(origin.denominator, cell_size.denominator)


def _axis_index_span(
    *,
    origin: int,
    cell_size: int,
    count: int,
    lo: int,
    hi: int,
) -> tuple[int, int] | None:
    """返回本轴上与区间 [lo, hi) 有正长度交叠的格索引半开区间 [first, last)。

    只纳入内部真正相交的格；仅在边界上相切（交叠长度为 0）的格不计入。
    无交叠时返回 None。坐标与格宽均为同一整数刻度下的整数，
    因此取整是精确的，不存在浮点边界误差。
    """
    if lo >= hi:
        return None

    # first = 包含 lo 的格索引（floor）；last = 首个起点不小于 hi 的格索引（ceil，排他）。
    first = (lo - origin) // cell_size
    last = -((origin - hi) // cell_size)  # ceil((hi - origin) / cell_size)

    if first < 0:
        first = 0
    if last > count:
        last = count
    if first >= last:
        return None
    return first, last


def build_overlap(source: GridMetrics, target: GridMetrics) -> OverlapTable:
    """构造源、目标两张格网的交叠表。

    实现策略（避免全量两两试算）：

    1. 先求两张格网整体 AABB 的交集 [lo_x, hi_x) × [lo_y, hi_y)；
    2. 把整体交集映射到两张格网各自的行、列索引范围；
    3. 仅在这些索引范围内部枚举格对，并用实际几何求交叠面积。

    因此时间复杂度只与“实际相交的格对数量”及其边界条带相关，
    与两张格网总格数的乘积无关。

    几何计算把十进制米值转成整数刻度上的精确整数（各轴公共分母），
    枚举时只做整数运算，面积最后才还原成“整数面积 / 公共面积分母”。
    """
    s_ox = source.exact_origin_x
    s_oy = source.exact_origin_y
    s_w = source.exact_cell_width
    s_h = source.exact_cell_height
    t_ox = target.exact_origin_x
    t_oy = target.exact_origin_y
    t_w = target.exact_cell_width
    t_h = target.exact_cell_height

    s_den_x = _axis_denominator(s_ox, s_w)
    s_den_y = _axis_denominator(s_oy, s_h)
    t_den_x = _axis_denominator(t_ox, t_w)
    t_den_y = _axis_denominator(t_oy, t_h)

    # 两张格网在同一轴上用同一个公共分母，便于直接比较与求交。
    den_x = math.lcm(s_den_x, t_den_x)
    den_y = math.lcm(s_den_y, t_den_y)
    s_mul_x, t_mul_x = den_x // s_den_x, den_x // t_den_x
    s_mul_y, t_mul_y = den_y // s_den_y, den_y // t_den_y

    s_ox_i = int(s_ox * s_den_x) * s_mul_x
    s_oy_i = int(s_oy * s_den_y) * s_mul_y
    s_w_i = int(s_w * s_den_x) * s_mul_x
    s_h_i = int(s_h * s_den_y) * s_mul_y
    t_ox_i = int(t_ox * t_den_x) * t_mul_x
    t_oy_i = int(t_oy * t_den_y) * t_mul_y
    t_w_i = int(t_w * t_den_x) * t_mul_x
    t_h_i = int(t_h * t_den_y) * t_mul_y

    s_max_x_i = s_ox_i + source.cols * s_w_i
    s_max_y_i = s_oy_i + source.rows * s_h_i
    t_max_x_i = t_ox_i + target.cols * t_w_i
    t_max_y_i = t_oy_i + target.rows * t_h_i

    lo_x = max(s_ox_i, t_ox_i)
    hi_x = min(s_max_x_i, t_max_x_i)
    lo_y = max(s_oy_i, t_oy_i)
    hi_y = min(s_max_y_i, t_max_y_i)

    area_denominator = den_x * den_y
    # 单格面积（整数刻度）：宽以 den_x 为分母、高以 den_y 为分母。
    source_cell_area_units = s_w_i * s_h_i
    target_cell_area_units = t_w_i * t_h_i

    if lo_x >= hi_x or lo_y >= hi_y:
        return OverlapTable(
            source=source,
            target=target,
            pairs=(),
            area_denominator=area_denominator,
            source_cell_area_units=source_cell_area_units,
            target_cell_area_units=target_cell_area_units,
        )

    source_cols = _axis_index_span(
        origin=s_ox_i, cell_size=s_w_i, count=source.cols, lo=lo_x, hi=hi_x
    )
    target_cols = _axis_index_span(
        origin=t_ox_i, cell_size=t_w_i, count=target.cols, lo=lo_x, hi=hi_x
    )
    source_rows = _axis_index_span(
        origin=s_oy_i, cell_size=s_h_i, count=source.rows, lo=lo_y, hi=hi_y
    )
    target_rows = _axis_index_span(
        origin=t_oy_i, cell_size=t_h_i, count=target.rows, lo=lo_y, hi=hi_y
    )

    if source_cols is None or target_cols is None:
        return OverlapTable(
            source=source,
            target=target,
            pairs=(),
            area_denominator=area_denominator,
            source_cell_area_units=source_cell_area_units,
            target_cell_area_units=target_cell_area_units,
        )
    if source_rows is None or target_rows is None:
        return OverlapTable(
            source=source,
            target=target,
            pairs=(),
            area_denominator=area_denominator,
            source_cell_area_units=source_cell_area_units,
            target_cell_area_units=target_cell_area_units,
        )

    t_col_lo, t_col_hi = target_cols
    t_row_lo, t_row_hi = target_rows

    pairs: list[OverlapPair] = []

    for tr in range(t_row_lo, t_row_hi):
        t_y0 = t_oy_i + tr * t_h_i
        t_y1 = t_y0 + t_h_i

        # 该目标行与整体交集的公共段，以及对应的源行范围。
        row_lo = max(t_y0, lo_y)
        row_hi = min(t_y1, hi_y)
        row_span = _axis_index_span(
            origin=s_oy_i, cell_size=s_h_i, count=source.rows, lo=row_lo, hi=row_hi
        )
        if row_span is None:
            continue

        for tc in range(t_col_lo, t_col_hi):
            t_x0 = t_ox_i + tc * t_w_i
            t_x1 = t_x0 + t_w_i

            col_lo = max(t_x0, lo_x)
            col_hi = min(t_x1, hi_x)
            col_span = _axis_index_span(
                origin=s_ox_i, cell_size=s_w_i, count=source.cols, lo=col_lo, hi=col_hi
            )
            if col_span is None:
                continue

            for sr in range(row_span[0], row_span[1]):
                s_y0 = s_oy_i + sr * s_h_i
                overlap_h = min(t_y1, s_y0 + s_h_i) - max(t_y0, s_y0)
                if overlap_h <= 0:
                    continue
                for sc in range(col_span[0], col_span[1]):
                    s_x0 = s_ox_i + sc * s_w_i
                    overlap_w = min(t_x1, s_x0 + s_w_i) - max(t_x0, s_x0)
                    if overlap_w <= 0:
                        continue
                    pairs.append(
                        OverlapPair(
                            target_row=tr,
                            target_col=tc,
                            source_row=sr,
                            source_col=sc,
                            area_units=overlap_w * overlap_h,
                            area_denominator=area_denominator,
                        )
                    )

    return OverlapTable(
        source=source,
        target=target,
        pairs=tuple(pairs),
        area_denominator=area_denominator,
        source_cell_area_units=source_cell_area_units,
        target_cell_area_units=target_cell_area_units,
    )
