# 설계 노트 — `models/feasibility.py` (Phase 5, Stage 2 추진 가능성)

> 규칙1: 코드 전에 이 노트. 승인 후 구현. CLAUDE.md §5(R4·R18)·§6 우선.

## 0. ★기대치 먼저 (이 노트의 심장 — 규칙9 정신)

**Stage 2 v1은 "무산 확률"을 내지 않는다.** 못 내기 때문이다:
- **R18 예측 천장**: 선정·무산은 주민동의율·정치·시장 사이클 등 *필지 피처에 없는 외생변수*가
  좌우한다. 완벽한 데이터로도 못 넘는 근본 천장.
- **데이터 실측**: hard negative(해제="요건 됐는데 무산")가 **학습 24노드뿐**(labels §13).
  Phase 3에서 B1·GNN 모두 hard-neg FPR 0.58 — "낡았는데 왜 안 됐나"를 *못 배웠다*.

→ **정직한 v1 산출 = ① calibration된 랭킹("상위 X%") + ② 리스크 신호 자리표시.** 단정적
"이 구역 무산 확률 30%"는 R18·R15 위반. 진단/예언 분리(§6)의 '예언' 쪽 → 천장 명시.

## 1. 이 모듈이 하는 일

심장1(B1)이 낸 "재개발 환경 적합도" 원점수를 **잘 보정(calibration)된 확률 + 백분위 랭킹**으로
바꿔 "상위 X%" 형태로 제시한다. 여기에 **무산 리스크는 v1 미학습**임을 명시하고, 미래 신호
(재추진 사례 등)를 받을 **자리표시**를 둔다. 학습 대상은 Phase 1 (필지,시점) 라벨.

### ★출력 명칭 정직성 (점수 축소한 만큼 이름도 축소 — 한 몸)
이 점수의 정체는 **"재개발 환경 유사도"**지 "추진 성공 가능성"이 *아니다*(B1이 배운 게
전자). → 사용자 표시 문구 = **"재개발 환경 점수 상위 X%"** (❌"추진 가능성 상위 X%"). 출력
메타에 박는다: `"선정·추진 성공 여부는 외생변수(주민동의·정치) 영역으로 본 점수가 측정하지
않음 (R18)"`. 이름이 점수보다 크면 그 자체가 R15 위반(과대 약속).

## 2. 왜 calibration인가 (랭킹의 정직성)

B1이 내는 raw 확률은 순서는 유의미해도 *수치 자체*는 과/소신될 수 있다(예: 0.9가 실제 90%
아님). "상위 X%"·"확률"로 사용자에게 보이려면 **보정된 확률**이어야 정직하다.
- **isotonic/Platt calibration**을 ★**공간 CV(LODO) out-of-fold 예측 위에서** 적합(R3 — train
  확률로 calibration하면 과적합·누수). reliability curve로 보정 품질 측정.
- 출력은 보정확률 + 백분위 랭킹. "확률"보다 **랭킹("상위 X%")을 1차 표현**으로(R18 천장 존중).

## 3. 라벨 (R4 — reliable_neg vs uncertain)

Phase 1 (필지,시점) 테이블. ★**uncertain(노후미지정)은 calibration 학습에서 제외**(R4 — "아직
모름"을 음성으로 쓰면 편향). positive vs reliable_negative로 보정. hard(해제)/easy(신축) 분리
리포트(hard n=24 → "통계적 무의미" 라벨, labels §13).

## 4. 리스크 신호 자리표시 (R18 — v1 미학습, 미래 슬롯)

"무산 리스크"는 v1 학습 불가 → **명시적 자리표시**로 둔다(거짓 0/단정 금지):
- 출력에 `risk_signals: {status: "v1_미학습", reason: "외생변수 R18 + 해제 n=24"}`.
- 미래(v1.1+) 신호 슬롯: 재추진 이력(장위13 사례)·주민동의율·조합 분쟁 뉴스(nlp layer3) 등.
- ★단정 회피가 면책(R15)과 한 몸 — 리스크를 "신호"로만, "확률"로 단정 안 함.

## 5. 평가 (정확도 금지)

- **calibration 신뢰도**: reliability diagram(예측확률 구간별 실제 양성률) + ECE(기대보정오차).
- **랭킹 품질**: 상위 랭킹이 positive를 얼마나 앞에 모으나(PR-AUC·상위 K 정밀도). 정확도 아님.
- **hard/easy 분리**: 해제(n=24 무의미)·신축 따로. 외생변수 천장을 수치로 보이기.

## 6. 함수/파일 분해

```text
models/feasibility.py
├── oof_scores(aug, edge_index, pnu_to_idx, *, cfg) -> Series   # B1 LODO out-of-fold 보정용 원점수
├── calibrate(scores, y, *, method='isotonic') -> calibrator    # 공간 OOF 위 적합(누수 차단)
├── feasibility_rank(calibrated_prob) -> percentile             # "상위 X%"
├── risk_signal_placeholder() -> dict                           # R18 v1 미학습 명시
└── score_feasibility(aug, ...) -> {calibrated_prob, rank_pct, risk_signals, caveats}
```

심장1 B1을 재사용(R9 일관) — feasibility는 그 위의 *보정·랭킹·정직성 레이어*다(새 모델 아님).

## 7. ★수검 (규칙9, 구현 후)

1. **calibration 신뢰도**: reliability diagram이 대각선에 가까운가(ECE 보고). 보정 전/후 비교.
2. **누수 차단 증명**: calibration이 *OOF* 예측 위에서 적합됐는지(train 확률 미사용).
3. **hard/easy 분리 출력**: 해제(n=24 무의미 라벨)·신축 따로 — 외생변수 천장 가시화.
4. **랭킹 sanity**: 상위 X% 구역이 실제 positive를 앞에 모으나(상위 K 정밀도).

## 8. 검토했지만 버린 대안

- **"무산 확률" 단정 출력**: R18·R15 위반, 데이터(n=24)로 불가능 → 랭킹+자리표시.
- **uncertain을 음성으로 학습**: R4 위반(우측절단 편향) → 제외.
- **새 feasibility 모델 별도 학습**: hard-neg 24개로 무의미 → B1 재사용+보정 레이어.
- **train 확률로 calibration**: 누수·과적합 → 공간 OOF 위에서만.
