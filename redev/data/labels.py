"""labels.py — (필지, 시점) 시점 라벨 테이블 (Phase 1 키스톤, §5).

역할: 데이터 파이프라인의 첫 산출물이자 최고 레버리지. 한 행 = "PNU p를 시점 t의
상태로 본 한 장면". 이 (필지,시점) 구조가 R1(노후도 시점누수)·R4(우측절단)·
R5(해제→재지정)·R11(분합필)을 동시에 푼다. ★여전히 정적 노드 분류(동적 GNN 아님).

조립: positive=ZoneTable∩parcels(t는 zone_boundary가 끝냄) / reliable_neg=해제seed
+신축파생 / uncertain=노후미지정 + R2 오염플래그 + 충돌해소. 설계: docs/design/labels.md.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from redev.config import load_thresholds
from redev.data.aging import old_ratio_by_parcel

# 라벨 테이블 스키마(§3). 모든 소스가 이 컬럼으로 정렬돼 합쳐진다.
LABEL_COLUMNS = [
    "pnu", "t", "t_date", "t_source", "t_alt", "label", "certainty",
    "source", "neg_reason", "zone_id", "zone_type", "contaminated",
]
# 충돌 우선순위(같은 (pnu,t)에서): positive > reliable_negative > uncertain.
_CERTAINTY_PRIORITY = {"positive": 0, "reliable_negative": 1, "uncertain": 2}


def _positives_from_zonetable(zone_table: gpd.GeoDataFrame, parcels: gpd.GeoDataFrame) -> pd.DataFrame:
    """ZoneTable(폴리곤+t) ∩ parcels 대표점 → positive rows.

    t-해소는 zone_boundary가 끝냈다(여기선 ∩만). 한 필지가 겹치는 두 구역에
    걸리면 **더 이른 t**(earliest = R1 보수)로 1행만 남긴다.
    """
    # 대표점 within 폴리곤: 경계 중복 더블카운트를 피하려 centroid 아닌 대표점.
    pts = gpd.GeoDataFrame(
        parcels[["pnu"]].copy(),
        geometry=parcels.geometry.representative_point(),
        crs=parcels.crs,
    )
    j = gpd.sjoin(
        pts,
        zone_table[["zone_id", "t", "t_date", "t_source", "t_alt", "zone_type", "geometry"]],
        predicate="within", how="inner",
    )
    j = j.sort_values("t").drop_duplicates("pnu", keep="first")   # 겹치면 earliest t
    return pd.DataFrame({
        "pnu": j["pnu"].values, "t": j["t"].values, "t_date": j["t_date"].values,
        "t_source": j["t_source"].values, "t_alt": j["t_alt"].values,
        "label": 1, "certainty": "positive", "source": "의제처리", "neg_reason": pd.NA,
        "zone_id": j["zone_id"].values, "zone_type": j["zone_type"].values, "contaminated": False,
    })


def _flag_contamination(pos: pd.DataFrame, buildings: pd.DataFrame, th: dict, *, current_year: int) -> pd.DataFrame:
    """R2(★구역레벨 *현재* 노후도 기준): positive 구역의 현재 노후도<컷이면 철거→신축
    완료(완공) → 그 구역 필지 전부 contaminated=True. (Phase 1은 플래그만, drop은 학습 전.)

    ★왜 현재인가(수검 교정): 필지 as-of-t<컷은 "완공"이 아니라 "t시점 안 낡은 1990년대
    필지"를 과대플래그한다(3,442필지). 완공구역은 옛 건물 철거로 as-of-t가 NaN이라 오히려
    *놓친다*(응암2 백련산: as-of-2008=NaN, 현재=0.00). 현재 노후도가 올바른 완공 탐지기.
    설계: labels.md §2-2. 생존편향(완공 성공구역 누락)은 §13 + v1.1 말소대장.
    """
    cut = th["label_hygiene"]["min_old_ratio_for_positive"]
    flagged = pos.copy()
    now = old_ratio_by_parcel(
        buildings[buildings["pnu"].isin(set(pos["pnu"]))], current_year, weight="area"
    )
    flagged["_now"] = flagged["pnu"].map(now)
    zone_now = flagged.groupby("zone_id")["_now"].mean()        # 구역레벨 현재 노후도
    completed = set(zone_now[zone_now < cut].index)             # 완공 구역
    flagged["contaminated"] = flagged["zone_id"].isin(completed)
    return flagged.drop(columns="_now")


def _derived_neg_uncertain(
    parcels: gpd.GeoDataFrame, buildings: pd.DataFrame, positive_pnus: set,
    *, current_year: int, th: dict,
) -> pd.DataFrame:
    """신축파생(reliable_neg) + 노후미지정(uncertain) — building_gis 파생, t=현재(§4-4).

    as-of-현재 노후도: <newbuild.max_old_ratio → 신축파생, ≥undesignated.min_old_ratio
    → 노후미지정. 사이 모호대역(0.2~0.5)·positive·노후도 결측은 제외.
    """
    nb_cut = th["newbuild"]["max_old_ratio"]
    ud_cut = th["undesignated"]["min_old_ratio"]
    ratio = old_ratio_by_parcel(buildings, current_year, weight="area")
    ratio = ratio[~ratio.index.isin(positive_pnus)].dropna()      # positive·평가불가 제외
    rows = []
    nb = ratio[ratio < nb_cut]
    ud = ratio[ratio >= ud_cut]
    for pnu in nb.index:
        rows.append(_neg_row(pnu, current_year, "reliable_negative", "new_construction", "신축파생"))
    for pnu in ud.index:
        rows.append(_neg_row(pnu, current_year, "uncertain", "not_yet", "노후미지정"))
    return pd.DataFrame(rows, columns=LABEL_COLUMNS) if rows else pd.DataFrame(columns=LABEL_COLUMNS)


def _neg_row(pnu, t, certainty, neg_reason, source) -> dict:
    return {
        "pnu": pnu, "t": int(t), "t_date": f"{int(t)}-01-01", "t_source": "derived_current",
        "t_alt": pd.NA, "label": 0, "certainty": certainty, "source": source,
        "neg_reason": neg_reason, "zone_id": pd.NA, "zone_type": pd.NA, "contaminated": False,
    }


def _cancelled_to_rows(cancelled_df: pd.DataFrame) -> pd.DataFrame:
    """cancelled.load_cancelled 출력 → 라벨 스키마 정렬(reliable_neg)."""
    df = cancelled_df.copy()
    for c in ("t_alt", "zone_type"):
        if c not in df.columns:
            df[c] = pd.NA
    if "contaminated" not in df.columns:
        df["contaminated"] = False
    return df.reindex(columns=LABEL_COLUMNS)


def _resolve_conflicts(rows: pd.DataFrame) -> pd.DataFrame:
    """같은 (pnu, t)에 여러 소스 → 우선순위 1행. ★다른 t는 별개 행 유지(R5)."""
    rows = rows.copy()
    rows["_prio"] = rows["certainty"].map(_CERTAINTY_PRIORITY).fillna(9)
    rows = rows.sort_values("_prio").drop_duplicates(["pnu", "t"], keep="first")
    return rows.drop(columns="_prio").reset_index(drop=True)


def build_label_table(
    zone_table: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    buildings: pd.DataFrame,
    cancelled_df: pd.DataFrame,
    *,
    current_year: int = 2026,
) -> tuple[pd.DataFrame, dict]:
    """(필지, 시점) 라벨 테이블 조립 — Phase 1 키스톤.

    입력: zone_table(zone_boundary), parcels(geom 스파인), buildings(정제 건물),
          cancelled_df(cancelled.load_cancelled). current_year=파생 neg의 t.
    출력: 라벨 DataFrame(LABEL_COLUMNS) + stats 리포트.

    R5: 같은 PNU가 해제(t1,neg)+지정(t2,pos)이면 다른 t라 자동 두 행. R11: positive·
    파생은 현재 parcels에서만 나오므로 현재필지 보장; 해제 미매칭은 cancelled가 이미 drop.
    """
    th = load_thresholds()
    pos = _positives_from_zonetable(zone_table, parcels)
    pos = _flag_contamination(pos, buildings, th, current_year=current_year)
    positive_pnus = set(pos["pnu"])

    canc = _cancelled_to_rows(cancelled_df)
    derived = _derived_neg_uncertain(
        parcels, buildings, positive_pnus, current_year=current_year, th=th
    )

    table = pd.concat([pos, canc, derived], ignore_index=True)
    table = _resolve_conflicts(table)

    cert = table["certainty"].value_counts().to_dict()
    report = {
        "total_rows": len(table),
        "certainty_counts": cert,
        "positive_zones": int(pos["zone_id"].nunique()),
        "contaminated_positives": int(table["contaminated"].sum()),
        "cancelled_neg": int((table["source"] == "해제").sum()),
        "newbuild_neg": int((table["source"] == "신축파생").sum()),
        "uncertain": int((table["source"] == "노후미지정").sum()),
        "r5_double_pnus": int(table["pnu"].duplicated(keep=False).sum()),  # 같은 PNU 다른 t
    }
    return table, report
