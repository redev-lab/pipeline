"""land_price.py — 개별공시지가(연도별 2001~2026) 적재 + as-of-t 백분위 피처 (v1.1, 가치 축).

설계: docs/design/features_v1_1.md §1-1. ★시점정합(R1): 학습 피처는 라벨 t의 연도값(as-of-t),
추론은 2026. ★(연도, 자치구) 백분위 상대화(labels §9) — 시대·지역 시세차 오염 완화. 결측은
백분위 0.5 + 결측 플래그(결측 자체를 신호로 보존). 조인키 = 토지코드(=PNU 19자리).
"""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_YEARLY_DIR = Path("_data/raw/추가데이터/공시지가 연도별")
_PRICE_COL = "공시지가(원/㎡)"


@lru_cache(maxsize=None)
def load_land_price(year: int) -> pd.DataFrame:
    """한 해 개별공시지가 → [pnu, price]. 토지코드=PNU(19자리), 19자리 아닌 행 drop(검수)."""
    z = zipfile.ZipFile(_YEARLY_DIR / f"공시지가_{year}년.zip")
    raw = z.read(f"공시지가_{year}년.csv")
    df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding="cp949", usecols=["토지코드", _PRICE_COL])
    df = df[df["토지코드"].str.len() == 19].copy()
    df["price"] = pd.to_numeric(df[_PRICE_COL], errors="coerce")
    # 같은 PNU 중복 행(데이터 아티팩트) → 1행(map 인덱스 유일성). 마지막값 채택.
    df = df.dropna(subset=["price"]).drop_duplicates("토지코드", keep="last")
    return df[["토지코드", "price"]].rename(columns={"토지코드": "pnu"})


def _year_percentile(year: int) -> pd.Series:
    """그 해 (자치구) 백분위 — pnu→pct(0~1). 자치구(PNU 앞5) 내 가격 순위."""
    df = load_land_price(year).copy()
    df["gu"] = df["pnu"].str[:5]
    df["pct"] = df.groupby("gu")["price"].rank(pct=True)        # (year, 구) 백분위(labels §9)
    return df.set_index("pnu")["pct"]


def land_price_features(label_rows: pd.DataFrame, *, current_year: int = 2026) -> pd.DataFrame:
    """(pnu, t) → as-of-t 공시지가 백분위 + 결측 플래그. 학습=t연도, 추론은 t=current_year.

    반환: [land_pct(0~1, 결측 0.5), land_missing(1=결측)]. 행 순서=label_rows.
    """
    rows = label_rows[["pnu", "t"]].reset_index(drop=True)
    pct = pd.Series(0.5, index=rows.index)
    missing = pd.Series(1, index=rows.index)
    for t, grp in rows.groupby("t"):
        yr = int(t) if int(t) <= current_year else current_year
        ymap = _year_percentile(yr)
        vals = grp["pnu"].map(ymap)
        pct.loc[grp.index] = vals.fillna(0.5).values
        missing.loc[grp.index] = vals.isna().astype(int).values
    return pd.DataFrame({"land_pct": pct.values, "land_missing": missing.values})
