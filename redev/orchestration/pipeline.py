"""pipeline.py — run(address): 파이프라인 등뼈 (Phase 6, §8 plain 파이썬). 설계: pipeline.md §3.

주소→PNU→클러스터→stage1(요건)→avm(시세맥락)→feasibility(환경점수)→eligibility(진입)를 직선
호출한다. ★LangGraph 아님(§8) — 직선 DAG. 동적 지점 둘: B1 저신뢰 폴백 if + ⑨ LLM 자리(Phase 7).
각 단계 try/except로 부분 실패(거래 0건 등)에도 전체가 죽지 않게(서비스 견고성).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from redev.data.location import admin_to_legal_dong, parse_location


@dataclass
class Context:
    """run()이 재사용하는 적재된 상태(한 번 build, 여러 주소). 추론은 캐시 조회라 빠름."""
    parcels: object
    buildings: object
    pnu_to_idx: dict
    edge_index: object
    jibun_index: dict
    scores: np.ndarray            # 전 노드 production B1 확률(전역 idx)
    calibrated: np.ndarray        # 보정확률(전역 idx) — feasibility 랭킹 참조분포
    pnu_cluster: dict             # pnu → 후보 클러스터(PNU 집합) 또는 None
    thr: float                    # 운영 임계값(OOF best-F1)
    target: pd.Series             # pnu → 대지지분 평당가
    agg_level: pd.Series          # pnu → 신뢰도 단계
    comp: pd.Series               # pnu → 비교신축 전용 평당가
    name2code: dict               # 구명 → 시군구코드


def build_context() -> Context:
    """캐시·모델을 한 번 적재해 Context 구성(여러 주소 재사용). 추론은 이후 조회라 빠름."""
    from redev.config import load_infer_config, training_districts
    from redev.data.ingest.parcels import build_jibun_index
    from redev.eval.metrics import best_f1
    from redev.models.avm import build_target, comparable_newbuild
    from redev.models.baseline import (_load_parcels_buildings, load_training_matrix,
                                       prepare_baseline_matrix)
    from redev.models.feasibility import calibrate, oof_scores
    from redev.models.infer import (candidate_clusters, operating_threshold, score_all,
                                    train_production_b1)

    tm = load_training_matrix()
    aug = prepare_baseline_matrix()
    parcels, buildings = _load_parcels_buildings()
    allf = pd.read_parquet("_data/processed/infer_features.parquet")
    trades = pd.read_parquet("_data/processed/_trades_36m.parquet")
    cfg = load_infer_config()

    oof = oof_scores(aug, tm.edge_index, tm.pnu_to_idx)
    y = aug["y"].to_numpy()
    msk = np.isfinite(oof)
    _, thr = best_f1(y[msk], oof[msk])
    cal = calibrate(oof, y)
    model, fc = train_production_b1(aug)
    scores = score_all(model, allf, fc)
    calibrated = cal.predict(scores)
    clusters = candidate_clusters(scores, tm.pnu_to_idx, tm.edge_index,
                                  thr=thr, min_nodes=cfg["cluster"]["min_nodes"])
    pnu_cluster = {p: cl for cl in clusters for p in cl}
    tgt = build_target(parcels, trades, current_ym="202606").set_index("pnu")
    comp = comparable_newbuild(parcels, trades).set_index("pnu")["comp_pyung"]
    name2code = {d["name"]: d["sigungu_code"] for d in training_districts()}
    return Context(parcels, buildings, tm.pnu_to_idx, tm.edge_index, build_jibun_index(parcels),
                   scores, calibrated, pnu_cluster, float(thr),
                   tgt["target_pyung"], tgt["agg_level"], comp, name2code)


def address_to_pnu(address: str, ctx: Context) -> str:
    """★4구 지번주소 → PNU. 도로명주소는 미지원(친절한 에러). PNU 직접입력도 허용(19자리)."""
    s = str(address).strip()
    if re.fullmatch(r"\d{19}", s):                      # PNU 직접입력
        return s
    s = re.sub(r"^서울(특별시)?\s*", "", s)
    gu = re.search(r"([가-힣]+구)", s)
    if not gu or gu.group(1) not in ctx.name2code:
        raise ValueError(f"4구(성북·동작·은평·구로) 밖이거나 구 미인식: {address}")
    rest = s[gu.end():].strip()
    if re.search(r"(로|길)\s*\d", rest) and not re.search(r"[동가]\s*\d", rest):
        raise ValueError("도로명주소 미지원 — v1은 지번주소만 (예: '성북구 정릉동 170-1').")
    parsed = parse_location(rest)
    if parsed is None:
        raise ValueError(f"지번 파싱 실패: '{rest}' (예: '정릉동 170-1')")
    dong, bon, bu, _san = parsed
    sig = ctx.name2code[gu.group(1)]
    pnu = ctx.jibun_index.get((sig, dong, bon, bu)) or ctx.jibun_index.get((sig, admin_to_legal_dong(dong), bon, bu))
    if pnu is None:
        raise ValueError(f"4구 내 해당 지번 없음: {address}")
    return pnu


def _stage(fn, *a, **k):
    """단계 try/except 래퍼 — 부분 실패도 전체 안 죽임. (status, value)."""
    try:
        return {"status": "ok", "result": fn(*a, **k)}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}


def run(address: str, ctx: Context, *, property_type: str | None = None, stage: str | None = None) -> dict:
    """주소 → 종합 판단(진단/예언 분리 + caveats). §8 직선 + 저신뢰 폴백 if + ⑨ 자리."""
    from redev.models.avm import market_context
    from redev.models.feasibility import score_feasibility
    from redev.rules.eligibility import score_eligibility
    from redev.rules.stage1 import score_cluster

    out: dict = {"input": address, "stages": {}}

    # ① 주소 → PNU
    try:
        pnu = address_to_pnu(address, ctx)
    except ValueError as e:
        return {"input": address, "error": str(e)}        # 입력 검증 실패는 즉시 반환
    out["pnu"] = pnu
    idx = ctx.pnu_to_idx.get(pnu)
    out["b1_score"] = round(float(ctx.scores[idx]), 3) if idx is not None else None

    # ★저신뢰 폴백 if (동적 지점 1): 점수 낮으면 "후보 환경 아님" 표시(그래도 시세는 보여줌)
    cluster = ctx.pnu_cluster.get(pnu)
    if idx is None or ctx.scores[idx] < ctx.thr or cluster is None:
        out["candidate"] = False
        out["note"] = "이 필지는 재개발 환경 후보 클러스터에 속하지 않음(저신뢰) — 시세 맥락만 제공."
    else:
        out["candidate"] = True
        out["cluster_size"] = len(cluster)
        # ④ stage1 요건(클러스터)
        out["stages"]["진단_요건"] = _stage(score_cluster, cluster, ctx.parcels, ctx.buildings)
        # ⑥ feasibility 환경 점수(클러스터 아니어도 노드 점수로 가능하나, 후보일 때만 의미)
        out["stages"]["예언_환경점수"] = _stage(
            score_feasibility, float(ctx.calibrated[idx]), ctx.calibrated)

    # ⑤ avm 시세 맥락(후보 여부 무관 — 진단)
    if pnu in ctx.target.index:
        out["stages"]["진단_시세맥락"] = _stage(
            market_context, float(ctx.target.get(pnu)),
            float(ctx.comp.get(pnu)) if pnu in ctx.comp.index and pd.notna(ctx.comp.get(pnu)) else float("nan"),
            agg_level=ctx.agg_level.get(pnu))
    else:
        out["stages"]["진단_시세맥락"] = {"status": "skipped", "reason": "반경 내 거래 0(타깃 결측)"}

    # ⑦ eligibility 진입(물건유형·단계 입력 있을 때)
    if property_type:
        out["stages"]["진입_eligibility"] = _stage(
            score_eligibility, property_type, stage or "조합설립인가")

    # ⑨ LLM 종합·설명 — ★Phase 7 자리(placeholder). v1은 구조화 dict.
    out["llm_summary"] = {"status": "v1_placeholder", "note": "종합·설명 LLM은 Phase 7."}
    out["caveats"] = [
        "v1 후보경계는 거친 필터(코어 ~39% 포착) — 정밀 경계 아님(R13).",
        "B1 점수는 '재개발 환경 유사도'(노후도 주도), 지정·추진과 강하게 정렬되진 않음(R4·R18).",
        "모든 수치 추정·참고치이며 투자 권유 아님(R15).",
    ]
    return out
