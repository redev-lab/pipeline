"""rail.py — 역세권 거리 (역사마스터) as-of-t 피처 (v1.1 ③ 입지 축).

설계: features_v1_1.md §1-3. WGS84(위경도)→EPSG:5186, 필지 centroid → 최근접 역 거리(cKDTree).
★시점정합: 학습 피처는 라벨 t에 ★개통한 노선의 역만(라벨창 2001~2025 내 개통 노선 필터, config
출처표 — 추정 아님). 역별 개통일 데이터셋 미발견 → 노선 단위 근사(open_year_source=compiled),
표에 없는 노선은 2001 이전(pre2001, 항상 포함). v2: 역별 개통일.
"""
from __future__ import annotations

from functools import lru_cache
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from redev.config import load_features_config
from redev.data.geo import TARGET_CRS
from redev.paths import DATA

_CSV = DATA / "raw/추가데이터/서울시 역사마스터 정보.csv"


@lru_cache(maxsize=None)
def load_stations() -> pd.DataFrame:
    """역사마스터 → [name, line, x, y(5186), open_year, open_source]. WGS84→5186 변환."""
    rail = load_features_config()["rail"]
    df = pd.read_csv(_CSV, dtype=str, encoding="cp949").rename(columns={"역사명": "name", "호선": "line"})
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
        df["경도"].astype(float), df["위도"].astype(float)), crs="EPSG:4326").to_crs(TARGET_CRS)
    df["x"], df["y"] = g.geometry.x.values, g.geometry.y.values
    lo = rail["line_open_year"]
    df["open_year"] = df["line"].map(lo).fillna(rail["default_pre_year"]).astype(int)
    df["open_source"] = np.where(df["line"].isin(lo), "compiled", "pre2001")   # ★출처 구분
    return df[["name", "line", "x", "y", "open_year", "open_source"]]


def rail_features(label_rows: pd.DataFrame, parcels, *, current_year: int = 2026, cfg=None,
                  centroids=None) -> pd.DataFrame:
    """(pnu, t) → as-of-t 최근접 역 거리(m) + 정규화 + 최근접역 출처. 행 순서=label_rows.

    학습: t에 개통한 노선 역만 최근접. 추론: current_year. ★누수 차단(2010 라벨이 2017 우이신설 못 봄).
    centroids: 사전계산된 pnu→centroid(Series) 주입 시 재계산 회피(이웃집계 반복 호출 최적화).
    """
    rail = (cfg or load_features_config())["rail"]
    st = load_stations()
    cent = centroids if centroids is not None else parcels.set_index("pnu").geometry.centroid
    rows = label_rows[["pnu", "t"]].reset_index(drop=True)
    cx = rows["pnu"].map(cent.x).to_numpy()
    cy = rows["pnu"].map(cent.y).to_numpy()
    dist = np.full(len(rows), np.nan)
    src = np.array(["none"] * len(rows), dtype=object)
    for t, grp in rows.groupby("t"):
        yr = min(int(t), current_year)
        elig = st[st["open_year"] <= yr]
        if elig.empty:
            continue
        tree = cKDTree(elig[["x", "y"]].to_numpy())
        gi = grp.index.to_numpy()
        d, idx = tree.query(np.c_[cx[gi], cy[gi]])
        dist[gi] = d
        src[gi] = elig["open_source"].to_numpy()[idx]
    # rail_prox: 역세권 근접도 0~1(1km 내 선형, 가까울수록 1) — 모델 친화 정규화.
    prox = 1.0 - np.clip(dist / rail["radius_norm_m"], 0, 1)
    return pd.DataFrame({"rail_dist_m": dist, "rail_prox": prox, "rail_src": src})
