"""zone_matching (분석 스크립트 호환용 re-export).

★canonical은 redev/data/zone_matching.py. 이 파일은 분석 스크립트가 기존 경로로
import하던 호환용일 뿐, 로직은 두지 않는다(두 벌이면 한쪽만 고쳐져 갈라진다 —
t-전쟁 버그 6개 로직은 한 곳에만 산다). 과거 검증본 스냅샷은 git d81a848.
"""
from redev.data.zone_matching import (  # noqa: F401
    CYCLE_DONE_KW,
    PROMOTION_PARENTS,
    REDEV_TITLE_KW,
    clean_text,
    cycle_done,
    is_redev_title,
    normalize_zone_name,
    parent_of,
    region_of,
)
