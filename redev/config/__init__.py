"""config — 설정 단일 입구 (narrow gate).

역할: districts.yaml / legal_thresholds.yaml 을 읽는 *유일한* 통로다.
다른 모듈은 YAML 경로도, yaml.safe_load 도 직접 만지지 않고 여기 함수만
호출한다. 이렇게 입구를 하나로 좁혀야 "자치구·임계값의 단일 출처"가 코드
구조로 강제된다(foundation.md 결정 B, CLAUDE.md 규칙7·5).

왜 입구를 좁히나: 여기저기서 YAML을 직접 로드하면 자치구 리스트의 출처가
분산되어, "서울 전역 확장 = 설정 한 줄 추가"가 깨진다. 설정을 읽는 곳이
하나여야 그 약속이 성립한다.
"""

from functools import lru_cache
from pathlib import Path

import yaml  # YAML 파서. safe_load = 임의 파이썬 객체 역직렬화를 막는 안전 로더.

# 이 패키지 디렉토리. YAML은 이 파일 옆에 산다 → 작업 디렉토리와 무관하게
# 항상 같은 위치에서 읽는다(상대경로 의존 제거).
_CONFIG_DIR = Path(__file__).resolve().parent
_DISTRICTS_PATH = _CONFIG_DIR / "districts.yaml"
_THRESHOLDS_PATH = _CONFIG_DIR / "legal_thresholds.yaml"
_GRAPH_PATH = _CONFIG_DIR / "graph.yaml"
_AVM_PATH = _CONFIG_DIR / "avm.yaml"
_ELIG_PATH = _CONFIG_DIR / "eligibility.yaml"
_INFER_PATH = _CONFIG_DIR / "infer.yaml"


def _load_yaml(path: Path) -> dict:
    """YAML 파일 하나를 파이썬 dict로 읽는다 (내부 헬퍼).

    safe_load: YAML을 dict/list/스칼라 같은 기본 타입으로만 역직렬화한다
    (임의 객체 생성 차단). 설정 파일엔 이걸로 충분하고 더 안전하다.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"빈 설정 파일이거나 파싱 실패: {path}")
    return data


@lru_cache(maxsize=None)
def load_graph_config() -> dict:
    """graph.yaml 파싱 (인접 그래프 구성: 제외지목·버퍼·밀도반경의 단일 출처)."""
    return _load_yaml(_GRAPH_PATH)


@lru_cache(maxsize=None)
def load_avm_config() -> dict:
    """avm.yaml 파싱 (심장2 AVM: 반경집계·비교신축·상승여력 시나리오의 단일 출처)."""
    return _load_yaml(_AVM_PATH)


@lru_cache(maxsize=None)
def load_eligibility_config() -> dict:
    """eligibility.yaml 파싱 (진입: 토허 분기·단계 잔여기간의 단일 출처. ★수시변경 규제)."""
    return _load_yaml(_ELIG_PATH)


@lru_cache(maxsize=None)
def load_infer_config() -> dict:
    """infer.yaml 파싱 (추론 출력: 클러스터 컷·폴리곤·백분위 히트맵의 단일 출처)."""
    return _load_yaml(_INFER_PATH)


@lru_cache(maxsize=None)
def load_districts() -> dict:
    """districts.yaml 전체를 파싱해 반환한다 (자치구 정보의 단일 출처).

    역할: "어느 구를 도는가"를 코드가 아니라 설정에서 받게 하는 입구.
    lru_cache로 파일은 프로세스당 한 번만 읽는다(설정은 런타임에 안 바뀐다).

    반환: {"districts": {"training": [...], "inference_extra": [...]}}
    """
    return _load_yaml(_DISTRICTS_PATH)


@lru_cache(maxsize=None)
def load_thresholds() -> dict:
    """legal_thresholds.yaml 전체를 파싱해 반환한다 (임계값의 단일 출처).

    역할: rules/stage1.py 등이 노후도·접도율 같은 법정 임계값을 코드 매직넘버
    대신 여기서 받게 하는 입구(CLAUDE.md 규칙5).
    """
    return _load_yaml(_THRESHOLDS_PATH)


def training_districts() -> list[dict]:
    """학습 대상 자치구 리스트 (각 항목: {name, sigungu_code}).

    역할: 모델 학습이 도는 구. 호출부가 dict 구조를 파고들지 않게 하는 헬퍼.
    """
    return load_districts()["districts"]["training"]


def inference_districts() -> list[dict]:
    """추론/데모 대상 자치구 = 학습 4구 + 추론 전용 추가(마포·강남).

    역할: 추론은 학습한 구 + 데모용 구까지 모두 커버한다. inductive GNN이면
    학습 안 한 추가 구에도 추론이 돌아야 한다(Phase 3).
    """
    d = load_districts()["districts"]
    return d["training"] + d["inference_extra"]


def training_sigungu_codes() -> set[str]:
    """학습 자치구의 시군구코드 집합 (PNU 필터용 편의 헬퍼).

    역할: redev.data.pnu.filter_by_districts 에 바로 넘길 수 있는 코드 집합.
    """
    return {d["sigungu_code"] for d in training_districts()}


def inference_sigungu_codes() -> set[str]:
    """추론 자치구(학습+추가)의 시군구코드 집합 (PNU 필터용 편의 헬퍼)."""
    return {d["sigungu_code"] for d in inference_districts()}
