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
from redev.paths import DATA


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
    zone_vectors: object = None   # 사례검색용 51구역 벡터(retrieval)
    pnu_zone: dict = None         # pnu → 지정구역 zone_id (고시 계획정보 조회용)
    zone_attrs: dict = None       # zone_id → 고시 계획정보(용적률·세대수 등, verified/flagged)


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
    allf = pd.read_parquet(DATA / "processed/infer_features.parquet")
    trades = pd.read_parquet(DATA / "processed/_trades_36m.parquet")
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

    # 사례검색용 51구역 벡터(retrieval) — ⑨ 리포트에서 "유사 구역"
    from redev.data.ingest.zone_boundary import load_zones
    from redev.data.labels import _positives_from_zonetable
    from redev.models.baseline import _RAW, _SRC, _vsizip
    from redev.retrieval.case_search import build_zone_vectors
    codes = sorted(name2code.values())
    zt, _ = load_zones(_vsizip(*_SRC["uq"]), str(_RAW / _SRC["gosi"]), parcels, codes,
                       jeonbisaeop_csv=str(_RAW / _SRC["jeonbisaeop"]),
                       shintong_csv=str(_RAW / _SRC["shintong"]),
                       public_redev_csv=str(_RAW / _SRC["public_redev"]))
    pos = _positives_from_zonetable(zt, parcels)
    ztype = zt.set_index("zone_id")["zone_type"].to_dict()
    zlist = [{"zone_id": z, "pnus": set(g["pnu"]), "t": int(g["t"].iloc[0]), "zone_type": ztype.get(z)}
             for z, g in pos.groupby("zone_id")]
    zv = build_zone_vectors(zlist, parcels, buildings)
    return Context(parcels, buildings, tm.pnu_to_idx, tm.edge_index, build_jibun_index(parcels),
                   scores, calibrated, pnu_cluster, float(thr),
                   tgt["target_pyung"], tgt["agg_level"], comp, name2code, zv)


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


def _confidence(score: float, thr: float, margin: float) -> str:
    """★신뢰도 — 운영임계값에서 margin 이상 떨어지면 '고신뢰'(확실히 높음/낮음), 근처면 '저신뢰'(애매).

    점수 최하위(0.058)가 '저신뢰'로 역전되던 버그 수정: 신뢰도는 '점수 높낮이'가 아니라
    '경계에서의 거리'다. 극단값일수록 분류가 확실 → 고신뢰.
    """
    return "고신뢰" if abs(score - thr) >= margin else "저신뢰"


def _stage(fn, *a, **k):
    """단계 try/except 래퍼 — 부분 실패도 전체 안 죽임. (status, value)."""
    try:
        return {"status": "ok", "result": fn(*a, **k)}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}


def _verdict(out: dict) -> dict:
    """★결정론 결론(계약 v1.1 §11-6) — 한 문장 headline + 행동분류 class. LLM 아님(규칙4).

    headline의 백분위는 '예언_환경점수' 표시값과 동일 원값(rank_top_pct)을 써 환각검증과 일치.
    분류: 후보(요건 경로) / 관심(점수 높으나 군집 밖) / 대상 아님(점수 낮음). 단정 회피 문구 고정.
    """
    from redev.config import load_infer_config
    fe = (out["stages"].get("예언_환경점수", {}) or {}).get("result") or {}
    pct = fe.get("rank_top_pct")
    pct_s = fe.get("rank_phrase") or "환경 점수 산출 불가"     # ★표시 문구(상위/하위, §B-1)와 일치
    if out.get("in_zone"):                                    # ★실제 지정 정비구역(의제처리)
        cls = "지정 정비구역"
        head = (f"★실제 지정 정비구역(정비계획 확정). 환경 점수 {pct_s}는 노후환경 상대순위(참고)일 뿐 "
                f"— 지정 여부와 무관. 추정·참고치, 단정 아님.")
    elif out.get("candidate"):                                # ★환경 유사 후보 — 우리 데이터 미지정(누락 가능)
        cls = "환경 유사 후보(미지정)"
        # ★'지정 아님' 단정 금지 — 우리 라벨(의제처리 재개발구역)은 부분집합이라 일반 주택재개발 누락 가능.
        head = (f"환경 유사 {pct_s} · 노후 환경이 닮은 후보 군집. "
                f"우리 데이터(의제처리 재개발구역) 기준 지정 구역으로 확인 안 됨 — 누락 가능, 직접 확인 권장. 단정 아님.")
    else:
        interest_pct = load_infer_config()["cluster"]["tight_top_pct"]   # 상위 N%면 '관심'(경계 밖)
        if pct is not None and pct <= interest_pct:
            cls = "관심(경계 밖)"
            head = f"환경 유사 {pct_s}이나 후보 군집 미포함 — 현 시점 관망. 우리 데이터 기준 미지정(누락 가능)."
        else:
            cls = "대상 아님"
            head = f"환경 유사 {pct_s} — 재개발 환경과 거리가 있어 현 시점 대상 아님. 단정 아님."
    return {"class": cls, "headline": head}


def run(address: str, ctx: Context, *, property_type: str | None = None, stage: str | None = None,
        with_report: bool = False) -> dict:
    """주소 → 종합 판단(진단/예언 분리 + caveats). §8 직선 + 저신뢰 폴백 if + ⑨ 리포트(opt-in)."""
    from redev.config import load_infer_config
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

    # ★저신뢰 폴백 if (동적 지점 1): 점수 낮거나 군집 미형성이면 "후보 아님"(그래도 시세·점수는 제공)
    cluster = ctx.pnu_cluster.get(pnu)
    candidate = not (idx is None or ctx.scores[idx] < ctx.thr or cluster is None)
    out["candidate"] = candidate

    # ★in_zone(실제 지정 정비구역) ≠ candidate(환경 유사 군집) — 혼동 금지(논현동 단계누수 수정).
    #   candidate = '노후 환경이 닮은 고점 군집'(지정 아님). in_zone = 실제 지정구역(pnu_zone, 의제처리).
    #   '언제'(잔여기간)·'지정' 판단은 in_zone일 때만. 환경 후보를 '지정됨'으로 오독하지 않게(§defect 1·2).
    in_zone = pnu in (getattr(ctx, "pnu_zone", None) or {})
    out["in_zone"] = in_zone

    # ★신뢰도 — 임계값에서 멀수록 고신뢰(확실히 높음/낮음), 근처면 저신뢰(애매). 점수-라벨 역전 방지.
    margin = load_infer_config()["cluster"]["confidence_margin"]
    out["confidence"] = "저신뢰" if idx is None else _confidence(float(ctx.scores[idx]), ctx.thr, margin)

    # ⑥ 환경 점수 — ★candidate 무관 항상 산출. 백분위는 raw 점수 순위(§B-2), 보정확률은 메타로 전달.
    if idx is not None:
        out["stages"]["예언_환경점수"] = _stage(
            score_feasibility, float(ctx.scores[idx]), ctx.scores,
            calibrated_prob=float(ctx.calibrated[idx]))
    else:
        out["stages"]["예언_환경점수"] = {"status": "na", "reason": "그래프 노드 외 — 환경 점수 산출 불가"}

    # ④ stage1 요건 — 클러스터(여러 필지)가 있어야 룰셋 가능. 없으면 사유 표기(계약 §11-2, "—" 금지)
    if candidate:
        out["cluster_size"] = len(cluster)
        out["stages"]["진단_요건"] = _stage(score_cluster, cluster, ctx.parcels, ctx.buildings)
    else:
        out["note"] = f"이 필지는 재개발 환경 후보 클러스터에 속하지 않음({out['confidence']}) — 시세 맥락만 제공."
        out["stages"]["진단_요건"] = {
            "status": "na", "reason": "후보 군집 미형성 — 단일 필지로 요건 판정 불가"}

    # ⑤ avm 시세 맥락(후보 여부 무관 — 진단)
    if pnu in ctx.target.index:
        out["stages"]["진단_시세맥락"] = _stage(
            market_context, float(ctx.target.get(pnu)),
            float(ctx.comp.get(pnu)) if pnu in ctx.comp.index and pd.notna(ctx.comp.get(pnu)) else float("nan"),
            agg_level=ctx.agg_level.get(pnu))
    else:
        out["stages"]["진단_시세맥락"] = {"status": "skipped", "reason": "반경 내 거래 0(타깃 결측)"}

    # 계획정보(고시 추출, §5) — 질의 필지가 지정구역이면 그 구역 용적률·세대수 등. verified만 단정.
    zid = (getattr(ctx, "pnu_zone", None) or {}).get(pnu)
    za = (getattr(ctx, "zone_attrs", None) or {}).get(zid) if zid else None
    if za and za.get("attrs"):
        out["stages"]["진단_계획정보"] = {"status": "ok", "result": za}

    # ⑦ eligibility 진입(물건유형 입력 시). ★잔여기간('언제')은 ★in_zone(실제 지정구역)일 때만 —
    #    candidate(환경 후보)면 단계 출력 금지(논현동 단계누수 수정). stage 기본값도 차단(계약 §11-3).
    if property_type:
        out["stages"]["진입_eligibility"] = _stage(
            score_eligibility, property_type, stage, in_zone=in_zone)

    out["caveats"] = [
        "v1 후보경계는 거친 필터(코어 ~39% 포착) — 정밀 경계 아님(R13).",
        "B1 점수는 '재개발 환경 유사도'(노후도 주도), 지정·추진과 강하게 정렬되진 않음(R4·R18).",
        "보존지구·상업지역 등은 점수가 높아도 정비 대상이 아닐 수 있음 — 용도지역 미반영(D-2 수검).",
        "모든 수치 추정·참고치이며 투자 권유 아님(R15).",
    ]
    if not in_zone:                                           # ★라벨 커버리지 한계(R7) — '지정 아님' 단정 금지
        out["caveats"].append(
            "지정 여부는 의제처리 재개발구역 데이터 기준 — 일반 주택재개발·가로주택 등 일부 지정구역은 누락 가능(R7).")
    out["verdict"] = _verdict(out)         # ★결정론 한 문장 결론 + 행동분류(계약 §11-6, 규칙4: LLM 아님)

    # ⑨ 종합 리포트 — ★opt-in(LLM 호출, 한도·속도). retrieval(유사구역)+social(사회신호)+report.
    if with_report:
        from redev.llm.report import generate_report
        from redev.nlp.layer3 import social_signals
        from redev.retrieval.case_search import search_cases
        if cluster and ctx.zone_vectors is not None:
            out["retrieval"] = _stage(search_cases, cluster, ctx.parcels, ctx.buildings, ctx.zone_vectors).get("result")
        matches = (out.get("retrieval") or {}).get("matches") or []
        out["social"] = social_signals(matches[0]["zone_id"] if matches else None)   # 데모: 대개 신호 없음
        out["report"] = generate_report(out)
    else:
        out["llm_summary"] = {"status": "opt_in", "note": "with_report=True로 ⑨ 리포트 생성."}
    return out
