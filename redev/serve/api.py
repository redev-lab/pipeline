"""api.py — pipeline export 인터페이스 (Phase 8 [0]). 설계: demo.md §1·§4.

★순수 파이썬 export — backend(FastAPI)가 import한다. HTTP·juso·로그는 backend 책임(경계).
의존 방향 backend→pipeline 단방향. report(주소→판단)·screen(스크리너)·build_serve_context(6구).
"""
from __future__ import annotations

import os
import pickle
import time

import pandas as pd

from redev.paths import DATA

_SCORES_CACHE = DATA / "processed/infer_scores.parquet"
_CTX_CACHE = DATA / "processed/serve_ctx.pkl"


def report_codes() -> list[str]:
    """★/report 컨텍스트 범위 — /screen(전역 25구 점수 캐시)과 분리된 *서브셋*.

    측정(C③): 이 머신은 8GB RAM(가용 ~1.2GB). 전역 25구 컨텍스트는 89만 필지 geometry +
    클러스터 dict + pickle 스파이크로 OOM(프로세스 강제종료 — R10 메모리 벽). 그래서 /report는
    데모 구(학습 4구 + 마포·강남) + 비학습 spot 1구(용산)로 한정한다. 지도 히트맵/스크리너의
    '전역 커버'는 점수 캐시가 담당. 전역 /report는 v2(PostGIS 온디맨드 서브그래프, R10).
    REDEV_REPORT_CODES=11290,11440,... 로 override 가능.
    """
    from redev.config import training_districts
    env = os.environ.get("REDEV_REPORT_CODES")
    if env:
        return sorted(c.strip() for c in env.split(",") if c.strip())
    base = {d["sigungu_code"] for d in training_districts()} | {"11440", "11680", "11170"}
    return sorted(base)


def load_scores() -> pd.DataFrame:
    """6구 전 노드 점수 캐시 + ★lon/lat(EPSG:5186→4326 reproject) — 지도 표시용."""
    df = pd.read_parquet(_SCORES_CACHE)
    if "lon" not in df.columns:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
        df["lon"], df["lat"] = tr.transform(df["cx"].to_numpy(), df["cy"].to_numpy())
    return df


def screen(scores: pd.DataFrame, *, gu: str | None = None, min_pct: float = 0.0,
           toheo: bool | None = None, top_k: int = 50) -> list:
    """★스크리너 — 점수 캐시 정렬·필터(거의 공짜). 구·점수백분위 필터 → 상위 필지 리스트.

    toheo 필터는 물건유형 의존이라 백엔드에서 eligibility로 후처리(여기선 구·점수만). 좌표 동봉(지도).
    """
    df = scores
    if gu:
        df = df[df["sigungu"] == gu]
    df = df[df["score_pct"] >= min_pct].nlargest(top_k, "score")
    cols = [c for c in ["pnu", "sigungu", "score", "score_pct", "lon", "lat"] if c in df.columns]
    return df[cols].to_dict("records")


def report(address: str, ctx, *, property_type: str | None = None, stage: str | None = None) -> dict:
    """주소 → 5종 판단 + 리포트(run 위임). ★backend가 ctx를 1회 build해 재사용."""
    from redev.orchestration.pipeline import run
    return run(address, ctx, property_type=property_type, stage=stage, with_report=True)


def load_serve_context(*, rebuild: bool = False, log=print):
    """★서버 기동 진입점 — 직렬화된 컨텍스트가 있으면 로드(빠름), 없으면 빌드 후 저장.

    전역(25구) 컨텍스트는 빌드가 무거우므로(graph·피처·시세·사례) ★사전 직렬화 한 번 → 이후
    기동은 pickle 로드만(C 측정: 기동 10분+ 회피). REDEV_DATA_DIR로 캐시 위치 이동 가능.
    """
    if not rebuild and _CTX_CACHE.exists():
        t = time.time()
        with open(_CTX_CACHE, "rb") as f:
            ctx = pickle.load(f)
        log(f"[ctx] 직렬화 로드 {_CTX_CACHE.stat().st_size/1e6:.0f}MB ({time.time()-t:.0f}s)")
        return ctx
    ctx = build_serve_context(log=log)
    try:                                                   # ★저장 실패(메모리 등)는 비치명 — 빌드된 ctx로 기동
        _CTX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CTX_CACHE, "wb") as f:
            pickle.dump(ctx, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"[ctx] 직렬화 저장 {_CTX_CACHE.stat().st_size/1e6:.0f}MB")
    except (MemoryError, OSError) as e:
        log(f"[ctx] 직렬화 저장 생략(비치명: {type(e).__name__}) — 빌드 ctx로 기동")
        _CTX_CACHE.unlink(missing_ok=True)                 # 손상 파일 제거
    return ctx


def build_serve_context(*, log=print):
    """데모 서버 컨텍스트 — ★전역(25구). production B1+ 동결, ★구별 조리 배치(메모리 구 1개로 바운드).

    pipeline.Context를 전역 데이터로 채워 run()이 임의 서울 주소를 처리하게 한다. edge_index는
    런타임 미사용이라 None(직렬화 경량화). graph 인접은 구간 엣지 0이라 per-gu가 전역과 동치.
    """
    import numpy as np
    from redev.config import inference_districts, load_infer_config
    from redev.data.ingest.building_gis import load_buildings
    from redev.data.ingest.parcels import build_jibun_index, load_parcels
    from redev.data.ingest.transactions import load_transactions
    from redev.data.ingest.zone_boundary import load_zones
    from redev.data.labels import _positives_from_zonetable
    from redev.eval.metrics import best_f1
    from redev.graph.build import build_graph
    from redev.models.avm import build_target, comparable_newbuild
    from redev.models.baseline import _RAW, _SRC, _vsizip, load_training_matrix, prepare_baseline_matrix
    from redev.models.feasibility import calibrate, oof_scores
    from redev.models.infer import build_all_node_features, candidate_clusters, score_all, train_production_b1
    from redev.orchestration.pipeline import Context
    from redev.retrieval.case_search import build_zone_vectors

    codes = report_codes()                                             # ★/report 서브셋(8GB 메모리 한계, R10)
    t0 = time.time()
    parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
    buildings, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)
    log(f"[load] {len(codes)}구(서브셋) parcels {len(parcels):,} ({time.time()-t0:.0f}s)")

    aug = prepare_baseline_matrix()                                   # 4구 라벨(학습 동결)
    model, fc = train_production_b1(aug)
    tm4 = load_training_matrix()
    oof = oof_scores(aug, tm4.edge_index, tm4.pnu_to_idx)
    y = aug["y"].to_numpy(); msk = np.isfinite(oof); _, thr = best_f1(y[msk], oof[msk])
    cal = calibrate(oof, y)
    cfg = load_infer_config()
    min_nodes = cfg["cluster"]["min_nodes"]

    pnu_to_idx, scores_list, pnu_cluster = {}, [], {}                 # ★구별 조리 → 전역 누적
    for code in codes:
        tc = time.time()
        gp = parcels[parcels["sigungu"] == code].copy()
        if gp.empty:
            continue
        graph, p2i, _ = build_graph(gp)
        allf = build_all_node_features(gp, buildings, p2i, graph.edge_index)
        sc = score_all(model, allf, fc)                              # local idx 정렬
        for cl in candidate_clusters(sc, p2i, graph.edge_index, thr=thr, min_nodes=min_nodes):
            for p in cl:
                pnu_cluster[p] = cl
        for pnu, s in zip(allf["pnu"], sc):                          # 전역 idx 부여
            pnu_to_idx[pnu] = len(scores_list); scores_list.append(float(s))
        log(f"  [{code}] {len(gp):,}필지 ({time.time()-tc:.0f}s)")
    scores = np.asarray(scores_list)
    calibrated = cal.predict(scores)
    log(f"[score] 전역 {len(scores):,}노드 / 클러스터필지 {len(pnu_cluster):,} ({time.time()-t0:.0f}s 누적)")

    months = [f"{y_}{mn:02d}" for y_ in (2024, 2025) for mn in range(1, 13)][:18]    # 시세(데모 범위)
    jibun_index = build_jibun_index(parcels)
    trades, _ = load_transactions(jibun_index, sigungu_codes=codes, months=months)
    tgt = build_target(parcels, trades, current_ym="202606").set_index("pnu")
    comp = comparable_newbuild(parcels, trades).set_index("pnu")["comp_pyung"]
    log(f"[avm] 거래 {len(trades):,} ({time.time()-t0:.0f}s 누적)")

    name2code = {d["name"]: d["sigungu_code"] for d in inference_districts()}   # 주소파싱용 전역 25구명
    zt, _ = load_zones(_vsizip(*_SRC["uq"]), str(_RAW / _SRC["gosi"]), parcels, codes,   # ★사례는 서브셋만
                       jeonbisaeop_csv=str(_RAW / _SRC["jeonbisaeop"]), shintong_csv=str(_RAW / _SRC["shintong"]),
                       public_redev_csv=str(_RAW / _SRC["public_redev"]))
    pos = _positives_from_zonetable(zt, parcels)
    ztype = zt.set_index("zone_id")["zone_type"].to_dict()
    zlist = [{"zone_id": z, "pnus": set(g["pnu"]), "t": int(g["t"].iloc[0]), "zone_type": ztype.get(z)}
             for z, g in pos.groupby("zone_id")]
    zv = build_zone_vectors(zlist, parcels, buildings)
    pnu_zone = dict(zip(pos["pnu"], pos["zone_id"]))         # 필지→지정구역(계획정보 조회, §5)
    from redev.data.zone_attrs import load_zone_attrs
    zone_attrs = load_zone_attrs()                          # zone_id→고시 계획정보(verified만 단정)
    log(f"[zones] 사례 {len(zlist)}구역 / 계획정보 {len(zone_attrs)}구역 ({time.time()-t0:.0f}s 누적)")

    return Context(parcels, buildings, pnu_to_idx, None, jibun_index,    # edge_index=None(런타임 미사용)
                   scores, calibrated, pnu_cluster, float(thr),
                   tgt["target_pyung"], tgt["agg_level"], comp, name2code, zv,
                   pnu_zone=pnu_zone, zone_attrs=zone_attrs)
