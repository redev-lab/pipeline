"""api.py — pipeline export 인터페이스 (Phase 8 [0]). 설계: demo.md §1·§4.

★순수 파이썬 export — backend(FastAPI)가 import한다. HTTP·juso·로그는 backend 책임(경계).
의존 방향 backend→pipeline 단방향. report(주소→판단)·screen(스크리너)·build_serve_context(6구).
"""
from __future__ import annotations

import pandas as pd

_SCORES_CACHE = "_data/processed/infer_scores_6gu.parquet"


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


def build_serve_context():
    """데모 서버 컨텍스트 — ★6구(마포·강남 포함). production B1+ 동결, 6구 graph·점수·시세·사례.

    pipeline.Context를 6구 데이터로 채워 run()이 추론 구 주소도 처리하게 한다. 서버 기동 시 1회.
    """
    import numpy as np
    from redev.config import inference_sigungu_codes, load_infer_config, training_districts
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

    codes = sorted(inference_sigungu_codes())                          # 6구
    parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
    buildings, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)
    graph, pnu_to_idx, _ = build_graph(parcels)
    allf = build_all_node_features(parcels, buildings, pnu_to_idx, graph.edge_index)

    aug = prepare_baseline_matrix()                                   # 4구 라벨(학습 동결)
    model, fc = train_production_b1(aug)
    scores = score_all(model, allf, fc)
    tm4 = load_training_matrix()
    oof = oof_scores(aug, tm4.edge_index, tm4.pnu_to_idx)
    y = aug["y"].to_numpy(); m = np.isfinite(oof); _, thr = best_f1(y[m], oof[m])
    cal = calibrate(oof, y)
    calibrated = cal.predict(scores)
    cfg = load_infer_config()
    clusters = candidate_clusters(scores, pnu_to_idx, graph.edge_index, thr=thr, min_nodes=cfg["cluster"]["min_nodes"])
    pnu_cluster = {p: cl for cl in clusters for p in cl}

    months = [f"{y_}{mn:02d}" for y_ in (2024, 2025) for mn in range(1, 13)][:18]    # 시세(데모 범위)
    trades, _ = load_transactions(build_jibun_index(parcels), sigungu_codes=codes, months=months)
    tgt = build_target(parcels, trades, current_ym="202606").set_index("pnu")
    comp = comparable_newbuild(parcels, trades).set_index("pnu")["comp_pyung"]

    name2code = {d["name"]: d["sigungu_code"] for d in training_districts()}
    name2code.update({"마포구": "11440", "강남구": "11680"})
    # 사례검색 51구역(학습 4구 지정) — 추론 구 후보지도 4구 사례와 비교
    zt, _ = load_zones(_vsizip(*_SRC["uq"]), str(_RAW / _SRC["gosi"]), parcels, sorted(name2code.values()),
                       jeonbisaeop_csv=str(_RAW / _SRC["jeonbisaeop"]), shintong_csv=str(_RAW / _SRC["shintong"]),
                       public_redev_csv=str(_RAW / _SRC["public_redev"]))
    pos = _positives_from_zonetable(zt, parcels)
    ztype = zt.set_index("zone_id")["zone_type"].to_dict()
    zlist = [{"zone_id": z, "pnus": set(g["pnu"]), "t": int(g["t"].iloc[0]), "zone_type": ztype.get(z)}
             for z, g in pos.groupby("zone_id")]
    zv = build_zone_vectors(zlist, parcels, buildings)

    return Context(parcels, buildings, pnu_to_idx, graph.edge_index, build_jibun_index(parcels),
                   scores, calibrated, pnu_cluster, float(thr),
                   tgt["target_pyung"], tgt["agg_level"], comp, name2code, zv)
