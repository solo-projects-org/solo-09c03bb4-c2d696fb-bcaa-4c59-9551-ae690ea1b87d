"""服务层：登记缓存、水量守恒、缺测处理。"""

from __future__ import annotations

import pytest

from app.schemas import GridSpec
from app.service import RegistrationNotFoundError, Registry


def spec(origin_x, origin_y, w, h, cols, rows) -> GridSpec:
    return GridSpec(
        origin_x=origin_x,
        origin_y=origin_y,
        cell_width=w,
        cell_height=h,
        cols=cols,
        rows=rows,
    )


SRC = spec(0.0, 0.0, 100.0, 100.0, 4, 3)
TGT = spec(60.0, 40.0, 120.0, 120.0, 3, 2)


def test_identical_registration_reuses_overlap():
    registry = Registry()
    first = registry.register(SRC, TGT)
    second = registry.register(SRC.model_copy(deep=True), TGT.model_copy(deep=True))
    assert first.registration_id == second.registration_id
    assert first.reused is False
    assert second.reused is True
    assert registry.count() == 1


def test_registration_reuses_geometry_with_float_roundoff():
    """同一组十进制米值经不同浮点运算得到时，仍应识别为同一几何并复用。"""
    registry = Registry()
    first = registry.register(
        spec(0.1 + 0.2, 0.0, 0.3, 0.7, 4, 3),
        TGT,
    )
    second = registry.register(spec(0.3, 0.0, 0.3, 0.7, 4, 3), TGT)
    assert second.registration_id == first.registration_id
    assert second.reused is True
    assert registry.count() == 1
    assert registry.get(first.registration_id).source.origin_x == 0.3


def test_geometry_change_creates_new_id_and_old_id_kept():
    registry = Registry()
    first = registry.register(SRC, TGT)

    changed_target = TGT.model_copy(update={"origin_x": 70.0})
    second = registry.register(SRC, changed_target)

    assert second.registration_id != first.registration_id
    assert second.reused is False
    assert registry.count() == 2

    # 旧登记号仍按原格网几何使用。
    old = registry.get(first.registration_id)
    assert old.target.origin_x == 60.0
    new = registry.get(second.registration_id)
    assert new.target.origin_x == 70.0


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("origin_x", 1.0),
        ("cell_width", 99.0),
        ("cell_height", 99.0),
        ("cols", 5),
        ("rows", 4),
    ],
)
def test_any_geometry_field_change_invalidates_cache(field, new_value):
    registry = Registry()
    first = registry.register(SRC, TGT)
    changed = SRC.model_copy(update={field: new_value})
    second = registry.register(changed, TGT)
    assert second.registration_id != first.registration_id


def test_transfer_conservation_with_full_coverage():
    """源范围完全被目标包住时，有记录水量应 100% 转移。"""
    # 源 2×2 格、每格 100m²；目标一个大格完全包住源。
    src = spec(0.0, 0.0, 10.0, 10.0, 2, 2)
    tgt = spec(-50.0, -50.0, 200.0, 200.0, 1, 1)
    registry = Registry()
    reg = registry.register(src, tgt)

    response = registry.transfer(
        reg.registration_id,
        [[10.0, 20.0], [30.0, None]],
    )
    assert len(response.results) == 1
    cell = response.results[0]
    # 有效面积 = 3 个有记录源格 × 100 m²（缺测格不计入）。
    assert cell.effective_area_m2 == pytest.approx(300.0)
    # 有效覆盖比例只统计有效面积，缺测格不算有效覆盖。
    assert cell.effective_coverage_ratio == pytest.approx(300.0 / 40_000.0)
    # 平均雨深为有记录源格的等权（等面积）平均。
    assert cell.mean_depth_mm == pytest.approx(20.0)
    # 体积 = 面积 m² × 雨深 mm × 1e-3。
    assert cell.effective_volume_m3 == pytest.approx(300.0 * 20.0 * 0.001)

    cons = response.conservation
    assert cons.recorded_source_volume_m3 == pytest.approx(
        100.0 * (10 + 20 + 30) * 0.001
    )
    assert cons.transferred_volume_m3 == pytest.approx(
        cons.recorded_source_volume_m3
    )
    assert cons.recorded_volume_outside_target_m3 == pytest.approx(0.0)
    assert cons.conservation_residual_m3 == pytest.approx(0.0, abs=1e-12)
    assert response.missing_source_cells == 1


def test_transfer_conservation_with_partial_coverage():
    """目标只覆盖源的一部分：目标外有记录水量须可核对、总量守恒。"""
    registry = Registry()
    reg = registry.register(SRC, TGT)
    # 全部有记录，统一 10 mm，便于手算。
    matrix = [[10.0 for _ in range(4)] for _ in range(3)]
    response = registry.transfer(reg.registration_id, matrix)

    reg_data = registry.get(reg.registration_id)
    overlap_area = reg_data.overlap.total_overlap_area
    source_area = 4 * 3 * 10_000.0

    cons = response.conservation
    # 统一雨深下，转移水量 = 交叠面积 × 10mm。
    assert cons.transferred_volume_m3 == pytest.approx(overlap_area * 10.0 * 0.001)
    assert cons.recorded_source_volume_m3 == pytest.approx(
        source_area * 10.0 * 0.001
    )
    assert cons.recorded_volume_outside_target_m3 == pytest.approx(
        (source_area - overlap_area) * 10.0 * 0.001
    )
    assert cons.conservation_residual_m3 == pytest.approx(0.0, abs=1e-10)

    # 各目标格有效体积之和等于总转移水量。
    assert sum(r.effective_volume_m3 for r in response.results) == pytest.approx(
        cons.transferred_volume_m3
    )
    # 统一雨深下每个有覆盖目标格的平均雨深仍是 10 mm。
    assert all(r.mean_depth_mm == pytest.approx(10.0) for r in response.results)


def test_missing_pattern_changes_day_to_day_without_reregistration():
    """同一登记号连续两天，缺测范围不同，结果随输入变化且无需重新登记。"""
    registry = Registry()
    reg = registry.register(SRC, TGT)

    day1 = [[10.0 for _ in range(4)] for _ in range(3)]
    day2 = [[10.0 for _ in range(4)] for _ in range(3)]
    day2[0][0] = None  # 仅第二天把一个源格标缺测

    r1 = registry.transfer(reg.registration_id, day1)
    r2 = registry.transfer(reg.registration_id, day2)

    assert r1.missing_source_cells == 0
    assert r2.missing_source_cells == 1

    # 缺测格若与某目标格交叠，第二天的有效面积与体积都应下降。
    reg_data = registry.get(reg.registration_id)
    missing_contrib_area = sum(
        p.area_m2
        for p in reg_data.overlap.pairs
        if p.source_row == 0 and p.source_col == 0
    )
    area_diff = sum(r.effective_area_m2 for r in r1.results) - sum(
        r.effective_area_m2 for r in r2.results
    )
    assert area_diff == pytest.approx(missing_contrib_area)

    vol_diff = r1.conservation.transferred_volume_m3 - (
        r2.conservation.transferred_volume_m3
    )
    assert vol_diff == pytest.approx(missing_contrib_area * 10.0 * 0.001)

    # 第二天的守恒仍然成立（缺测格直接不进入有记录水量）。
    assert r2.conservation.conservation_residual_m3 == pytest.approx(0.0, abs=1e-12)
    assert r2.conservation.recorded_source_volume_m3 == pytest.approx(
        r1.conservation.recorded_source_volume_m3 - 10_000.0 * 10.0 * 0.001
    )


def test_no_effective_cells_omits_depth_report():
    """目标格无有效覆盖时不报雨深；整张表无交叠时结果为空列表。"""
    disjoint_target = spec(900.0, 900.0, 120.0, 120.0, 3, 2)
    registry = Registry()
    reg = registry.register(SRC, disjoint_target)
    response = registry.transfer(
        reg.registration_id, [[10.0] * 4 for _ in range(3)]
    )
    assert response.results == []
    # 全部有记录水量都落在目标范围之外，守恒依旧成立。
    cons = response.conservation
    assert cons.transferred_volume_m3 == 0.0
    assert cons.recorded_volume_outside_target_m3 == pytest.approx(
        cons.recorded_source_volume_m3
    )

    # 全部缺测时即便几何上有交叠，也没有任何有效格。
    reg2 = registry.register(SRC, TGT)
    all_missing = registry.transfer(
        reg2.registration_id, [[None] * 4 for _ in range(3)]
    )
    assert all_missing.results == []
    assert all_missing.missing_source_cells == 12
    assert all_missing.conservation.recorded_source_volume_m3 == 0.0


def test_wrong_matrix_dimensions_rejected():
    registry = Registry()
    reg = registry.register(SRC, TGT)
    with pytest.raises(ValueError, match="维度应为"):
        registry.transfer(reg.registration_id, [[1.0, 2.0], [3.0, 4.0]])


def test_unknown_registration_raises_not_found():
    registry = Registry()
    with pytest.raises(RegistrationNotFoundError):
        registry.get("does-not-exist")


def test_area_weighted_mean_depth_with_varied_depths():
    """非均匀雨深：平均雨深必须按交叠面积加权，而非简单平均。"""
    # 一个目标格被两个等面积源格各覆盖一半（各 50 m²），雨深分别 10、30 mm。
    src = spec(0.0, 0.0, 10.0, 10.0, 2, 1)  # 两个 100 m² 源格
    tgt = spec(0.0, 0.0, 10.0, 10.0, 1, 1)  # 目标格只与源格 (0,0) 同尺寸
    registry = Registry()
    reg = registry.register(src, tgt)
    response = registry.transfer(reg.registration_id, [[10.0, 30.0]])
    cell = response.results[0]
    assert cell.effective_area_m2 == pytest.approx(100.0)
    assert cell.mean_depth_mm == pytest.approx(10.0)
    assert cell.effective_volume_m3 == pytest.approx(1.0)


def test_decimal_aligned_coverage_ratio_is_exactly_one():
    """小数米格网完全对齐时，覆盖比例应为 1，不能被 1.0000000000000002 拒绝。"""
    src = spec(0.1, 0.2, 0.3, 0.7, 4, 3)
    tgt = spec(0.1, 0.2, 0.3, 0.7, 4, 3)
    registry = Registry()
    reg = registry.register(src, tgt)
    response = registry.transfer(
        reg.registration_id, [[1.0] * 4 for _ in range(3)]
    )

    assert len(response.results) == 12
    assert all(cell.effective_coverage_ratio == 1.0 for cell in response.results)
    assert all(
        cell.effective_area_m2 == pytest.approx(0.3 * 0.7)
        for cell in response.results
    )
    assert response.target_cell_area_m2 == pytest.approx(0.21)

    cons = response.conservation
    assert cons.recorded_volume_outside_target_m3 == 0.0
    assert cons.conservation_residual_m3 == 0.0
    assert cons.transferred_volume_m3 == pytest.approx(
        cons.recorded_source_volume_m3
    )


def test_target_containing_source_reports_zero_outside_volume():
    """目标范围完全包住源格网时，目标外有记录水量必须精确为 0。"""
    src = spec(0.1, 0.2, 0.3, 0.7, 3, 3)
    tgt = spec(-1.0, -1.0, 10.0, 10.0, 1, 1)
    registry = Registry()
    reg = registry.register(src, tgt)
    response = registry.transfer(
        reg.registration_id, [[5.0] * 3 for _ in range(3)]
    )

    cons = response.conservation
    assert cons.recorded_volume_outside_target_m3 == 0.0
    assert cons.conservation_residual_m3 == 0.0
    assert cons.transferred_volume_m3 == pytest.approx(
        cons.recorded_source_volume_m3
    )
    cell = response.results[0]
    assert cell.effective_area_m2 == pytest.approx(9 * 0.3 * 0.7)
    assert cell.effective_coverage_ratio == pytest.approx(
        9 * 0.3 * 0.7 / 100.0
    )


def test_partial_coverage_decimal_strip_volume_is_counted_outside():
    """部分覆盖时，窄条雨量按“源格目标外面积 × 当天雨深”核算，不能被抹掉。"""
    # 源 10 列 × 1 行，每格 0.1 m × 0.7 m；目标只覆盖 x ∈ [0, 0.35)。
    src = spec(0.0, 0.0, 0.1, 0.7, 10, 1)
    tgt = spec(0.0, 0.0, 0.35, 0.7, 1, 1)
    registry = Registry()
    reg = registry.register(src, tgt)

    depth = 10.0
    response = registry.transfer(reg.registration_id, [[depth] * 10])
    cell = response.results[0]
    cons = response.conservation

    covered_area = 0.35 * 0.7
    total_area = 10 * 0.1 * 0.7
    # 目标格只收下 x[0, 0.35)，其中第 3 个源格只贡献 0.05 m 宽窄条。
    assert cell.effective_area_m2 == pytest.approx(covered_area)
    assert cell.effective_volume_m3 == pytest.approx(
        covered_area * depth * 0.001
    )
    # 目标外 = 每格源格落在目标外的面积 × 当天雨深逐格求和。
    assert cons.recorded_volume_outside_target_m3 == pytest.approx(
        (total_area - covered_area) * depth * 0.001
    )
    assert cons.recorded_source_volume_m3 == pytest.approx(
        total_area * depth * 0.001
    )
    assert cons.conservation_residual_m3 == 0.0


def test_partial_coverage_narrow_strip_keeps_its_own_depth():
    """窄条雨量按窄条自己的当天雨深核算，不能用平均雨深抹掉或摊平。"""
    # 源 4 列 × 1 行，每格 100 m × 100 m；目标只覆盖 x ∈ [0, 350)，
    # 第 3 个源格只剩 50 m 宽窄条落在目标内。
    src = spec(0.0, 0.0, 100.0, 100.0, 4, 1)
    tgt = spec(0.0, 0.0, 350.0, 100.0, 1, 1)
    registry = Registry()
    reg = registry.register(src, tgt)

    response = registry.transfer(reg.registration_id, [[10.0, 20.0, 30.0, 40.0]])
    cell = response.results[0]
    cons = response.conservation

    # 目标内：3 整格 + 第 3 格的 50 m 窄条，各自使用本格雨深。
    # 每整格面积 100 m × 100 m = 10 000 m²。
    expected_volume = (
        10_000 * 10 + 10_000 * 20 + 10_000 * 30 + 5_000 * 40
    ) * 0.001
    assert cell.effective_volume_m3 == pytest.approx(expected_volume)
    assert cell.effective_area_m2 == pytest.approx(35_000.0)
    # 窄条没有被抹掉，也没有把整格 40 mm 都算进来。
    assert cell.mean_depth_mm == pytest.approx(
        (10_000 * 10 + 10_000 * 20 + 10_000 * 30 + 5_000 * 40) / 35_000.0
    )
    # 目标外只有第 3 格剩下的 50 m 窄条，按该格 40 mm 核算。
    assert cons.recorded_volume_outside_target_m3 == pytest.approx(
        5_000 * 40 * 0.001
    )
    assert cons.conservation_residual_m3 == 0.0


def test_target_cell_covered_by_multiple_partial_sources_reports_one():
    """目标格被多个源格拼合完全覆盖（每格只是部分交叠）时，比例报 1。"""
    # 源 10 列 × 0.1 m 宽；目标格 x[0.15, 0.85) 完全落在源范围内，
    # 但由 7 个源格的完整或部分交叠拼成。
    src = spec(0.0, 0.0, 0.1, 0.5, 10, 1)
    tgt = spec(0.15, 0.0, 0.7, 0.5, 1, 1)
    registry = Registry()
    reg = registry.register(src, tgt)
    response = registry.transfer(reg.registration_id, [[10.0] * 10])

    cell = response.results[0]
    assert cell.effective_coverage_ratio == 1.0
    assert cell.effective_area_m2 == pytest.approx(0.7 * 0.5)
    assert cell.effective_volume_m3 == pytest.approx(0.7 * 0.5 * 10 * 0.001)

    # 目标格完全被覆盖，但源格网左右两侧仍落在目标外，守恒必须分清。
    cons = response.conservation
    assert cons.recorded_volume_outside_target_m3 == pytest.approx(
        (10 * 0.1 - 0.7) * 0.5 * 10 * 0.001
    )
    assert cons.conservation_residual_m3 == 0.0


def test_equally_shifting_both_grids_does_not_change_results():
    """两张格网一起平移后，各目标格覆盖、雨量与守恒明细都不应变化。"""
    base_src = dict(origin_x=0.1, origin_y=0.2, cell_width=0.3, cell_height=0.7, cols=10, rows=10)
    base_tgt = dict(origin_x=0.15, origin_y=0.25, cell_width=0.3, cell_height=0.7, cols=10, rows=10)
    matrix = [
        [float((r * 10 + c) % 7) for c in range(10)] for r in range(10)
    ]

    def run(src_kwargs, tgt_kwargs):
        registry = Registry()
        reg = registry.register(
            GridSpec(**src_kwargs), GridSpec(**tgt_kwargs)
        )
        return registry.transfer(reg.registration_id, matrix)

    baseline = run(base_src, base_tgt)

    for shift in (0.1, 0.3, 0.7, 1.1, 12.34):
        src = {**base_src, "origin_x": base_src["origin_x"] + shift,
               "origin_y": base_src["origin_y"] + shift}
        tgt = {**base_tgt, "origin_x": base_tgt["origin_x"] + shift,
               "origin_y": base_tgt["origin_y"] + shift}
        shifted = run(src, tgt)

        assert [c.model_dump() for c in shifted.results] == [
            c.model_dump() for c in baseline.results
        ], f"平移 {shift} 后目标格结果变化"
        assert shifted.conservation == baseline.conservation, (
            f"平移 {shift} 后守恒明细变化"
        )
