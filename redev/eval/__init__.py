"""eval — 검증 규약 (CLAUDE.md §7).

역할: 모델 성능을 정직하게 검증한다.
- spatial_cv.py: ★구역/자치구 단위 hold-out. 무작위 CV 금지(공간 누수 →
  같은 동네가 train/test에 섞여 성능 과대평가).
- leakage_ablation.py: 시점 의존 피처 포함/제외 대조.

(아직 비어 있음.)
"""
