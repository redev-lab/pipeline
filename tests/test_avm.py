"""avm 회귀 테스트 — 대지지분 평당가·계층폴백·시세맥락(순수, 합성).

실행: python -m pytest tests/test_avm.py
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from redev.models.avm import _PYUNG_M2, _villa_pyung, build_target, market_context


def test_villa_pyung_landshare():
    """대지지분 평당가 = 거래금액 / (대지권면적/3.3058). 대지권 유효한 villa만."""
    tr = pd.DataFrame([
        {"trade_type": "villa", "deal_amount": 33000, "land_share_m2": 30.0},
        {"trade_type": "villa", "deal_amount": 50000, "land_share_m2": None},  # 대지권 결측 제외
        {"trade_type": "apt", "deal_amount": 90000, "land_share_m2": 5.0},      # apt 제외
    ])
    v = _villa_pyung(tr)
    assert len(v) == 1
    assert abs(v["pyung"].iloc[0] - 33000 / (30.0 / _PYUNG_M2)) < 1e-6


def test_market_context_no_subtraction():
    """★시세 맥락: 두 평당가 병렬, 빼지 않음(단위 다름). 상승여력 수치 없음."""
    mc = market_context(5407, 4959, agg_level="r50", n_trades=8)
    assert mc["land_share_pyung_man"] == 5407 and mc["newbuild_exclu_pyung_man"] == 4959
    assert "upside" not in mc and "upside_band" not in mc      # 상승여력 수치 미산출
    assert any("빼지 않는다" in mc["note"] for _ in [0])


def test_build_target_fallback_levels():
    """가까운 거래 있는 필지=r50, 없는(같은 동) 필지=dong 폴백."""
    par = gpd.GeoDataFrame(
        [{"pnu": "1129010100100010000", "geometry": box(0, 0, 10, 10)},
         {"pnu": "1129010100100020000", "geometry": box(1000, 0, 1010, 10)}],
        geometry="geometry", crs="EPSG:5186",
    )
    tr = pd.DataFrame([{"pnu": "1129010100100010000", "trade_type": "villa",
                        "deal_amount": 33000, "land_share_m2": 30.0, "deal_ym": "202505"}])
    tg = build_target(par, tr, current_ym="202506").set_index("pnu")
    assert tg.loc["1129010100100010000", "agg_level"] == "r50"     # 거래 50m 내
    assert tg.loc["1129010100100020000", "agg_level"] == "dong"    # 1km 밖 → 동 폴백
    # 둘 다 같은 동이라 동 median으로 같은 값
    assert abs(tg.loc["1129010100100020000", "target_pyung"]
               - tg.loc["1129010100100010000", "target_pyung"]) < 1e-6
