"""avm.py — 심장2 Stage3 입지가치 (대지지분 평당단가 AVM). 설계: docs/design/avm.md.

★R6: 개별 호수 매핑 금지 → 필지 반경 집계 대지지분 평당단가가 타깃. 계층 폴백(50m→100m→동)
으로 거짓0(결측을 0으로)도 결측천지도 둘 다 피하고, 어느 단계서 채웠는지 agg_level로 기록.
★R14·R17 상승여력은 범위·시나리오로만(단정 금지). ★R16 거래편향·R15 규제역설 출력 명시.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from redev.config import load_avm_config

_PYUNG_M2 = 3.305785          # 1평(㎡). 대지지분 평당단가 = 거래금액 / (대지권면적/이 값)


def _villa_pyung(trades: pd.DataFrame) -> pd.DataFrame:
    """연립다세대 거래 → 대지지분 평당단가(만원/평). 대지권 유효한 것만(R6)."""
    v = trades[(trades["trade_type"] == "villa")
               & trades["land_share_m2"].notna() & (trades["land_share_m2"] > 0)].copy()
    v["pyung"] = v["deal_amount"] / (v["land_share_m2"] / _PYUNG_M2)
    return v


def _window_filter(trades: pd.DataFrame, current_ym: str, years: int) -> pd.DataFrame:
    """최근 N년 거래만(시세 최신성). current_ym='YYYYMM'."""
    cy, cm = int(current_ym[:4]), int(current_ym[4:6])
    cutoff = (cy - years) * 100 + cm
    ym = pd.to_numeric(trades["deal_ym"], errors="coerce")
    return trades[ym >= cutoff]


def build_target(parcels, trades, *, current_ym: str, cfg=None) -> pd.DataFrame:
    """★R6 타깃: 필지별 반경 집계 대지지분 평당단가(median) + agg_level. 계층 폴백.

    입력: parcels(geom — 거래·타깃 필지 좌표), trades(transactions 정제), current_ym.
    출력: [pnu, target_pyung, agg_level('r50'/'r100'/'dong'/'missing'), n_trades].
    """
    cfg = (cfg or load_avm_config())["target"]
    v = _window_filter(_villa_pyung(trades), current_ym, cfg["window_years"])

    # 거래 좌표 = 거래 필지 centroid (집합건물·호수 무관, 땅 위치 — R6)
    cent = parcels.set_index("pnu").geometry.centroid
    v = v.assign(x=v["pnu"].map(cent.x), y=v["pnu"].map(cent.y)).dropna(subset=["x", "y"])
    v["dong"] = v["pnu"].str[:10]                       # 법정동코드 10자리(폴백 단위)
    dong_med = v.groupby("dong")["pyung"].median()
    dong_cnt = v.groupby("dong")["pyung"].size()        # ★동 평균의 표본 거래수(출처 표기용)

    # 타깃 필지 centroid
    pc = parcels[["pnu"]].copy()
    c = parcels.geometry.centroid
    pc["x"], pc["y"], pc["dong"] = c.x.values, c.y.values, parcels["pnu"].str[:10].values

    n = len(pc)
    target = np.full(n, np.nan)
    level = np.array(["missing"] * n, dtype=object)
    ntr = np.zeros(n, dtype=int)
    tree = cKDTree(v[["x", "y"]].to_numpy())
    prices = v["pyung"].to_numpy()
    min_tr = cfg["min_trades"]

    for radius in cfg["radius_levels_m"]:                # 50 → 100 (가까운 단계 우선)
        todo = np.where(level == "missing")[0]
        if todo.size == 0:
            break
        nbrs = tree.query_ball_point(pc.iloc[todo][["x", "y"]].to_numpy(), r=radius)
        for k, idx in zip(todo, nbrs):
            if len(idx) >= min_tr:
                target[k] = np.median(prices[idx])
                level[k] = f"r{radius}"
                ntr[k] = len(idx)

    # 동 단위 폴백 (반경서 못 채운 필지) — ★재개발 구역은 거래 동결로 폴백 흔함
    todo = np.where(level == "missing")[0]
    dmap = pc["dong"].map(dong_med).to_numpy()
    cmap = pc["dong"].map(dong_cnt).to_numpy()
    for k in todo:
        if not np.isnan(dmap[k]):
            target[k] = dmap[k]
            level[k] = "dong"
            ntr[k] = int(cmap[k])                       # ★동 평균 표본수 기록(신뢰도 표기)

    return pd.DataFrame({"pnu": pc["pnu"].values, "target_pyung": target,
                         "agg_level": level, "n_trades": ntr})


# AVM 피처. v1=앞 6 + ★v1.1 가치/입지 축(공시지가 백분위·용도지역·역세권). agg_ord·n_trades=R16.
AVM_FEATURES = ["area_m2", "compactness", "road_abut", "aging", "n_trades", "agg_ord",
                "land_pct", "zoning_ord", "rail_prox"]
_AGG_ORD = {"r50": 0, "r100": 1, "dong": 2, "missing": 3}


def avm_features(parcels, buildings, target_df, *, current_year: int = 2026) -> pd.DataFrame:
    """target_df에 필지 피처(현재 노후도·면적·형상·접도) + 거래파생(신뢰도) 부착."""
    from redev.data.aging import old_ratio_by_parcel
    from redev.graph.features import _compactness, _road_abutting

    ps = parcels[parcels["pnu"].isin(set(target_df["pnu"]))].copy()
    ps["area_m2"] = ps.geometry.area
    ps["compactness"] = ps.geometry.map(_compactness)
    ps["road_abut"] = _road_abutting(ps, parcels[parcels["jimok"] == "도"]).values
    aging = old_ratio_by_parcel(buildings, current_year, weight="area")
    feat = ps.set_index("pnu")[["area_m2", "compactness", "road_abut"]]
    feat["aging"] = feat.index.map(aging).fillna(0.0)
    out = target_df.merge(feat, left_on="pnu", right_index=True, how="left")
    out["agg_ord"] = out["agg_level"].map(_AGG_ORD)
    # ★v1.1 가치/입지 축: 공시지가 백분위·용도지역 ordinal·역세권 근접(현재시점).
    from redev.data.ingest.land_price import land_price_features
    from redev.data.ingest.rail import rail_features
    from redev.data.ingest.zoning import zoning_features
    rows = pd.DataFrame({"pnu": out["pnu"].values, "t": current_year})
    out["land_pct"] = land_price_features(rows, current_year=current_year)["land_pct"].values
    zf = zoning_features(parcels).set_index("pnu")["zoning_ord"]
    out["zoning_ord"] = out["pnu"].map(zf).fillna(0.0).values
    out["rail_prox"] = rail_features(rows, parcels, current_year=current_year)["rail_prox"].values
    return out


def fit_avm(train_df: pd.DataFrame, *, params: dict | None = None):
    """★보조 모델(강등) — 결측 필지 보간 + SHAP 설명용. v1 메인 추정은 build_target 직접.

    ★측정 판정(2026-06-12, 시점분리): v1 피처(노후도·접도·형상)로는 이 부스팅이 *구평균
    베이스라인도 못 이긴다*(MAE 1707 vs 1565). 1·2위 피처가 n_trades·agg_ord(데이터 밀도
    프록시)라 — v1 피처는 "재개발 환경" 신호이지 "가치" 신호가 아니다. 가치 드라이버(역세권
    거리·용도지역·강남접근·학군)는 v1.1 → 그 입수 후 재평가. (심장1 GNN R9 판정과 동형: 병목은
    모델이 아니라 피처.) 따라서 v1 입지가치 출력 = build_target(반경집계)+agg_level 신뢰도.
    """
    import xgboost as xgb
    p = params or dict(max_depth=5, learning_rate=0.1, n_estimators=400,
                       tree_method="hist", n_jobs=-1, subsample=0.8, colsample_bytree=0.8)
    m = xgb.XGBRegressor(**p)
    m.fit(train_df[AVM_FEATURES], train_df["target_pyung"])
    return m


def comparable_newbuild(parcels, trades, *, current_year: int = 2026, cfg=None) -> pd.DataFrame:
    """★R17 비교신축 시세: 필지 반경 내 신축 아파트 전용 평당가(median). 기준 config 공개."""
    cfg = (cfg or load_avm_config())["comparable"]
    apt = trades[trades["trade_type"] == "apt"].copy()
    apt["pyung"] = apt["deal_amount"] / (apt["area_m2"] / _PYUNG_M2)            # 전용 평당가
    age = current_year - pd.to_numeric(apt["build_year"], errors="coerce")
    apt = apt[age <= cfg["max_age_years"]]                                     # 신축만
    cent = parcels.set_index("pnu").geometry.centroid
    apt = apt.assign(x=apt["pnu"].map(cent.x), y=apt["pnu"].map(cent.y)).dropna(subset=["x", "y"])
    pc = parcels[["pnu"]].copy()
    c = parcels.geometry.centroid
    pc["x"], pc["y"] = c.x.values, c.y.values
    tree = cKDTree(apt[["x", "y"]].to_numpy())
    prices = apt["pyung"].to_numpy()
    out = np.full(len(pc), np.nan)
    nout = np.zeros(len(pc), dtype=int)                                       # ★표본 거래수(출처 표기)
    for k, idx in enumerate(tree.query_ball_point(pc[["x", "y"]].to_numpy(), r=cfg["radius_m"])):
        if len(idx) >= cfg["min_trades"]:
            out[k] = np.median(prices[idx]); nout[k] = len(idx)
    return pd.DataFrame({"pnu": pc["pnu"].values, "comp_pyung": out, "n_trades": nout})


def _land_provenance(agg_level, n) -> str:
    """대지지분 시세 출처 문구 — 환경점수 caveat과 같은 수준으로 '무슨 값인지' 명시(오인 차단)."""
    n = int(n) if n else 0
    if agg_level == "r50":
        return f"반경 50m 실거래 {n}건"
    if agg_level == "r100":
        return f"반경 100m 실거래 {n}건"
    if agg_level == "dong":
        return f"주변 거래 부족 → 동 평균 {n}건"      # ★재개발 구역 동결 흔함 — '주변 시세' 오인 차단
    return "거래 없음"


def market_context(target_pyung: float, comp_pyung: float, *, agg_level=None,
                   n_trades=None, n_comp=None) -> dict:
    """★"시세 맥락"(상승여력 단정 회피, R14·R15). 두 사실을 *병렬* 제시 — 빼기 금지.

    ★측정 실증(2026-06-12): 대지지분 평당가 vs 전용 평당가는 단위가 달라 1:1로 빼면 음수
    천지로 오도된다. 그래서 '상승여력' 수치 대신 두 시세를 나란히 보여준다. ★섹션명을
    '상승여력'이 아니라 '시세 맥락'으로 — 단어가 숫자 하나를 기대하게 만들기 때문.

    상승여력 수식(신축시세 − 매입가 − 분담금)은 ★v1.1: 용적률·비례율·조합원분양가가 와야
    *정직하게* 계산된다('안 하는' 게 아니라 '재료 대기'). 수치 단정 안 함 → 면책 부담↓(§9).
    """
    def _n(x):                                       # ★NaN→None (JSON 직렬화 불가 차단 — comp 결측 등)
        return None if x is None or (isinstance(x, float) and x != x) else x
    nc = int(n_comp) if n_comp else 0
    return {
        "land_share_pyung_man": _n(target_pyung),   # 인근 빌라 대지지분 평당가(만원/평)
        "newbuild_exclu_pyung_man": _n(comp_pyung), # 인근 신축 아파트 전용 평당가(만원/평)
        "land_provenance": _land_provenance(agg_level, n_trades),          # ★대지지분 출처·표본
        "newbuild_provenance": (f"반경 1km 신축 {nc}건" if _n(comp_pyung) is not None and nc else "거래 부족"),
        "confidence": {"agg_level": agg_level, "n_target": _n(n_trades), "n_comp": _n(n_comp)},
        "note": "두 값은 단위가 다르다(대지지분 평당 vs 전용 평당) — 직접 빼지 않는다(시세 맥락).",
        "caveats": [
            "상승여력 수치는 v1 보류. 정직 계산엔 용적률·비례율·조합원분양가 필요(v1.1).",
            "추정·참고치이며 투자 권유 아님(R15) — 수치 단정을 안 하므로 면책 부담도 낮다.",
        ],
    }


def explain(model, X: pd.DataFrame) -> dict:
    """가치 기여(SHAP 있으면 SHAP, 없으면 gain 중요도). 설명가능성(심장2 정직성)."""
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X[AVM_FEATURES])
        return {"method": "shap", "mean_abs": dict(zip(AVM_FEATURES, np.abs(sv).mean(0).round(3)))}
    except Exception:
        imp = model.feature_importances_
        return {"method": "gain_importance", "importance": dict(zip(AVM_FEATURES, imp.round(3)))}
