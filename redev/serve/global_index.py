"""global_index.py — v1 부분 전역화 사이드카(주소 인덱스 + 전역 지정/후보). 타당성: docs/backlog/label_coverage.

7구 ctx 밖(예: 방배동/서초) 주소도 '환경 점수 + 판정(지정/후보)'만은 주기 위한 ★경량 사이드카.
- addr_index: (시군구,동,본,부) 해시 → PNU. compact int64(딕셔너리 816MB 회피 → 런타임 ~14MB·searchsorted).
- global_zone_cluster: PNU → in_zone(의제처리 25구)·candidate(후보 군집 25구).
★상세(시세·노후·접도·사례)는 7구 ctx 의존이라 전역 미제공 — 부분 리포트(정직).
오프라인 1회 빌드(per-gu 배치 → 메모리 1구 바운드, R10 회피). 런타임은 조회만.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from redev.paths import DATA

_ADDR = DATA / "processed/addr_index.parquet"          # key_hash(int64) → pnu(int64), key_hash 정렬
_ZC = DATA / "processed/global_zone_cluster.parquet"   # pnu(int64), in_zone(bool), cluster(bool)


def _key_hash(sig, dong, bon, bu) -> int:
    """(시군구,동,본,부) → 안정 int64 해시(blake2b 8B). 프로세스 무관 결정적(파이썬 hash 솔트 회피)."""
    h = hashlib.blake2b(f"{sig}|{dong}|{bon}|{bu}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=True)


def build_global_sidecars(*, force: bool = False, log=print) -> dict:
    """★오프라인 1회 — 25구 per-gu 배치로 addr_index + 전역 지정/후보 사이드카 생성(메모리 1구 바운드).

    cluster thr·min_nodes = build_serve_context와 동일(4구 학습 oof best_f1 / config). cluster 점수는
    기존 infer_scores 캐시 재사용(재스코어 안 함). ★per-gu 군집비율 분포 검증(25구 적용 상식성, 사용자 #2).
    """
    if not force and _ADDR.exists() and _ZC.exists():
        log("사이드카 이미 존재 — force=True로 재빌드")
        return {"addr": _ADDR, "zc": _ZC}
    import time

    from redev.config import inference_sigungu_codes, load_infer_config
    from redev.data.ingest.parcels import build_jibun_index, load_parcels
    from redev.data.ingest.zone_boundary import load_zones
    from redev.data.labels import _positives_from_zonetable
    from redev.eval.metrics import best_f1
    from redev.graph.build import build_graph
    from redev.models.baseline import _RAW, _SRC, _vsizip, load_training_matrix, prepare_baseline_matrix
    from redev.models.feasibility import oof_scores
    from redev.models.infer import candidate_clusters
    from redev.serve.infer_districts import build_inference_scores

    t0 = time.time()
    # cluster 임계 thr = 4구 학습 oof best_f1 (build_serve_context와 동일 기준)
    aug = prepare_baseline_matrix()
    tm4 = load_training_matrix()
    oof = oof_scores(aug, tm4.edge_index, tm4.pnu_to_idx)
    y = aug["y"].to_numpy()
    msk = np.isfinite(oof)
    _, thr = best_f1(y[msk], oof[msk])
    min_nodes = load_infer_config()["cluster"]["min_nodes"]
    log(f"[thr] cluster thr={thr:.3f} min_nodes={min_nodes}")

    cache = build_inference_scores()                       # 기존 25구 점수 캐시 재사용(재스코어 0)
    cache_score = dict(zip(cache["pnu"], cache["score"]))

    codes = sorted(inference_sigungu_codes())
    ak, ap = [], []
    zone_set, clus_set = set(), set()
    val = []
    for code in codes:                                     # ★per-gu(메모리 1구 바운드)
        tc = time.time()
        gp, _ = load_parcels(_vsizip(*_SRC["parcels"]), [code], with_geometry=True)
        if gp.empty:
            log(f"  [{code}] parcels 0 — 건너뜀"); continue
        for key, pnu in build_jibun_index(gp).items():     # 주소 인덱스(런타임과 동일 키)
            ak.append(_key_hash(*key)); ap.append(int(pnu))
        graph, p2i, _ = build_graph(gp)                    # 군집(캐시 점수 재사용)
        sc = np.zeros(len(p2i))
        for p, i in p2i.items():
            sc[i] = cache_score.get(p, 0.0)
        gclus = candidate_clusters(sc, p2i, graph.edge_index, thr=thr, min_nodes=min_nodes)
        cset = set().union(*gclus) if gclus else set()
        clus_set |= cset
        try:                                               # 지정(의제처리) — 지정 0인 구는 빈 GDF 크래시 → 0 처리
            zt, _ = load_zones(_vsizip(*_SRC["uq"]), str(_RAW / _SRC["gosi"]), gp, [code],
                               jeonbisaeop_csv=str(_RAW / _SRC["jeonbisaeop"]),
                               shintong_csv=str(_RAW / _SRC["shintong"]),
                               public_redev_csv=str(_RAW / _SRC["public_redev"]))
            zpos = set(_positives_from_zonetable(zt, gp)["pnu"]) if len(zt) else set()
        except (ValueError, KeyError):
            zpos = set()
        zone_set |= zpos
        ratio = len(cset) / max(1, len(gp))
        val.append((code, len(gp), len(gclus), len(cset), ratio, len(zpos)))
        log(f"  [{code}] 필지{len(gp):,} | 군집{len(gclus)}개·{len(cset)}필지({ratio:.1%}) | 지정{len(zpos)} ({time.time()-tc:.0f}s)")

    # ★검증(#2): 25구 군집필지 비율 분포 상식성 — 7구 기준을 전역 적용했을 때 이상치 탐지
    ratios = sorted(v[4] for v in val)
    log(f"[검증] 25구 군집필지비율 min{ratios[0]:.1%} 중앙{ratios[len(ratios)//2]:.1%} max{ratios[-1]:.1%}")
    out = [(v[0], f"{v[4]:.0%}") for v in val if v[4] > 0.6]
    if out:
        log(f"  ★이상치(군집비율>60% — 거의 전역이 후보=비상식): {out} → thr/min_nodes 재조정 검토")

    ak = np.asarray(ak, dtype=np.int64); ap = np.asarray(ap, dtype=np.int64)
    order = np.argsort(ak, kind="stable")                  # searchsorted용 정렬
    pd.DataFrame({"key_hash": ak[order], "pnu": ap[order]}).to_parquet(_ADDR, index=False)
    allp = sorted(zone_set | clus_set)
    pd.DataFrame({"pnu": [int(p) for p in allp],
                  "in_zone": [p in zone_set for p in allp],
                  "cluster": [p in clus_set for p in allp]}).to_parquet(_ZC, index=False)
    log(f"[done] addr {len(ak):,} · 지정 {len(zone_set):,} · 후보 {len(clus_set):,} / {time.time()-t0:.0f}s")
    return {"addr": _ADDR, "zc": _ZC, "validation": val}


def load_global_index() -> dict | None:
    """런타임 경량 로드 — addr(정렬 int64 배열 2개) + 지정/후보 PNU 집합. 사이드카 없으면 None."""
    if not (_ADDR.exists() and _ZC.exists()):
        return None
    a = pd.read_parquet(_ADDR)
    zc = pd.read_parquet(_ZC)
    return {
        "kh": a["key_hash"].to_numpy(np.int64),
        "pn": a["pnu"].to_numpy(np.int64),
        "zones": set(zc.loc[zc["in_zone"], "pnu"].astype("int64").tolist()),
        "clusters": set(zc.loc[zc["cluster"], "pnu"].astype("int64").tolist()),
    }


def lookup_pnu(gidx: dict, sig: str, dong: str, bon: int, bu: int) -> str | None:
    """(시군구,동,본,부) → PNU(19자리 문자열). searchsorted 조회(딕셔너리 없이 ~14MB)."""
    kh = _key_hash(sig, dong, bon, bu)
    arr = gidx["kh"]
    i = int(np.searchsorted(arr, kh))
    if i < len(arr) and int(arr[i]) == kh:
        return str(int(gidx["pn"][i])).zfill(19)
    return None
