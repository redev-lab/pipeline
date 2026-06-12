"""Stage1 수검 — 실제 지정구역 역검증(zone_type별·연대별 충족률 + 미달 전수 + 신축 negative).
분석 전용. → _data/processed/_stage1_accept.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.cluster import KMeans

from redev.config import training_sigungu_codes
from redev.data.ingest.building_gis import load_buildings
from redev.data.ingest.parcels import load_parcels
from redev.data.ingest.zone_boundary import load_zones
from redev.data.labels import _positives_from_zonetable
from redev.models.baseline import _RAW, _SRC, _vsizip, prepare_baseline_matrix
from redev.rules.stage1 import score_cluster

codes = training_sigungu_codes()
parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
buildings, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)
zones, _ = load_zones(_vsizip(*_SRC["uq"]), str(_RAW / _SRC["gosi"]), parcels, codes,
                      jeonbisaeop_csv=str(_RAW / _SRC["jeonbisaeop"]),
                      shintong_csv=str(_RAW / _SRC["shintong"]),
                      public_redev_csv=str(_RAW / _SRC["public_redev"]))
pos = _positives_from_zonetable(zones, parcels)
ztype = zones.set_index("zone_id")["zone_type"].to_dict()

rows = []
for zid, g in pos.groupby("zone_id"):
    out = score_cluster(g["pnu"].tolist(), parcels, buildings)
    m = out["metrics"]
    rows.append({"zone": zid, "ztype": ztype.get(zid, "?"), "t": int(g["t"].iloc[0]),
                 "n": m["n_parcels"], "path": out["path"],
                 "h": out["housing_eligible"], "u": out["urban_eligible"],
                 "old": m["old_area_ratio"], "abut": m["abut_ratio"],
                 "dens": m["house_density"], "under": m["undersized_ratio"], "low": m["low_density_ratio"]})

L = [f"지정구역 {len(rows)}개 (UQ1221 주택정비형 = 우리 룰 본진 / UQ1222 도시정비형)"]


def block(name, sub):
    if not sub:
        return
    redev = sum(r["path"] == "재개발" for r in sub)
    moa = sum(r["path"] == "모아타운·소규모정비" for r in sub)
    none = sum(r["path"] == "해당없음" for r in sub)
    L.append(f"  {name:<22} n={len(sub):>3}  재개발 {redev}({100*redev/len(sub):.0f}%)  "
             f"모아타운 {moa}  해당없음 {none}")


L.append("=== zone_type별 ===")
for zt in ("UQ1221", "UQ1222"):
    block(zt, [r for r in rows if r["ztype"] == zt])
L.append("=== 연대별(전체) ===")
for lo, hi, nm in [(0, 2010, "~2009"), (2010, 2020, "2010-19"), (2020, 9999, "2020+")]:
    block(nm, [r for r in rows if lo <= r["t"] < hi])
L.append("=== ★본진: UQ1221 연대별 ===")
for lo, hi, nm in [(0, 2010, "~2009"), (2010, 2020, "2010-19"), (2020, 9999, "2020+")]:
    block("UQ1221 " + nm, [r for r in rows if r["ztype"] == "UQ1221" and lo <= r["t"] < hi])

L.append("\n=== 미달 구역 전수(재개발 아님) — 어느 지표서 미달인지 ===")
for r in sorted([r for r in rows if r["path"] != "재개발"], key=lambda x: (x["ztype"], x["t"])):
    def f(x):
        return "NaN" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
    L.append(f"  {r['zone']} {r['ztype']} t{r['t']} n{r['n']} [{r['path']}] "
             f"노후{f(r['old'])} 접도{f(r['abut'])} 호밀{f(r['dens'])} 과소{f(r['under'])} 저밀{f(r['low'])}")

# ── 신축 negative 클러스터(4구 내) → 해당없음 확인 ──
aug = prepare_baseline_matrix()
nb = aug[(aug.y == 0) & (aug.neg_reason == "new_construction")]
km = KMeans(n_clusters=40, n_init=5, random_state=0).fit(nb[["centroid_x", "centroid_y"]].to_numpy())
L.append("\n=== 신축 negative 클러스터 2개 → 해당없음 기대 ===")
for b in (0, 1):
    pnus = nb.iloc[np.where(km.labels_ == b)[0]]["pnu"].tolist()[:150]
    o = score_cluster(pnus, parcels, buildings)
    L.append(f"  신축블록{b} n{len(pnus)} → [{o['path']}] 노후{o['metrics']['old_area_ratio']:.2f}")

open("_data/processed/_stage1_accept.txt", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print("DONE")
