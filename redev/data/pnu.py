"""pnu.py — PNU(필지 고유번호) 표준화·검증 (모든 데이터 조인의 단일 키).

역할: 출처가 제각각인 원천(건물GIS·실거래가·정비구역경계…)이 가진 필지
식별자를 *하나의 19자리 문자열*로 정규화한다. 이후 파이프라인의 모든 조인은
이 표준 PNU 위에서만 일어난다(CLAUDE.md 규칙6). 여기가 흔들리면 §5-3
(PNU 결측·깨짐)이 그대로 터진다.

★핵심 설계 — 방어선 2층 (docs/design/foundation.md 결정 A):
  PNU는 19자리(~1.1e18)로 float64 안전 정수 한계 2^53(~9.0e15)을 넘는다.
  PNU가 float로 들어온 시점엔 끝자리가 *이미* 깨져 있다 — 복구 불가능.
    1차(진짜 방어선): ingest에서 dtype=str 로 읽어 float이 될 기회를 차단.
    2차(이 파일): normalize_pnu 는 float을 받으면 coerce하지 않고 *거부*한다.
      float을 zero-pad하면 깨진 값을 "그럴듯한 가짜 PNU"로 둔갑시켜
      조인을 조용히 틀린 필지에 붙인다 — 크래시보다 나쁘다.

PNU 19자리 구조:
    [ 0:10] 법정동코드  = 시도2 + 시군구3 + 읍면동3 + 리2
    [10:11] 필지구분    = 1(일반/토지대장) · 2(산/임야대장)
    [11:15] 본번(4)
    [15:19] 부번(4)
    시군구코드(5) = [0:5]  → districts.yaml 의 sigungu_code 와 동일 키
"""

import numbers
from collections.abc import Iterable

# --- PNU 구조 상수 (매직넘버 대신 이름으로) ---------------------------------
PNU_LENGTH = 19
_LEGAL_DONG_CODE = slice(0, 10)   # 법정동코드 10자리
_SIGUNGU_CODE = slice(0, 5)       # 시군구코드 5자리 (시도2+시군구3)
_FILJI_GUBUN = 10                 # 필지구분 위치 (1=일반, 2=산)
_BONBEON = slice(11, 15)          # 본번 4자리
_BUBEON = slice(15, 19)           # 부번 4자리

_VALID_FILJI_GUBUN = {"1", "2"}   # 1=일반(토지대장), 2=산(임야대장)


def normalize_pnu(raw) -> str:
    """무엇이 들어와도(단, str·int만) 19자리 zero-pad 문자열 PNU로 표준화한다.

    역할: 모든 조인의 2차 방어선이자 표준화 관문. 표현이 제각각인 PNU
    (정수, 선행 0 빠진 짧은 문자열 등)를 단일 표준형으로 모은다.

    받는 것:
      - str: 공백 제거 후 숫자만 허용. 19자리 미만이면 선행 0으로 패딩.
      - int: 파이썬 int는 임의정밀도라 손실이 없다 → 허용. (numpy 정수 포함)
    거부하는 것:
      - float: ★이미 정밀도가 깨졌을 수 있다. coerce하면 가짜 PNU 양산 →
        TypeError. (해결: 호출부가 아니라 ingest에서 dtype=str로 읽어라.)
      - bool: int의 하위타입이지만 PNU로는 무의미 → 거부.

    반환: 길이 19의 숫자 문자열. (의미 유효성은 is_valid_pnu 로 별도 확인.)
    """
    # bool은 isinstance(_, int)에 걸리므로 float보다도 먼저 명시적으로 막는다.
    if isinstance(raw, bool):
        raise TypeError(f"PNU로 bool은 받지 않는다: {raw!r}")

    # ★float 거부 — 결정 A의 2차 방어선. numpy.float64도 float 하위타입이라 여기 걸린다.
    if isinstance(raw, float):
        raise TypeError(
            f"PNU를 float으로 받지 않는다(이미 정밀도 손실 가능): {raw!r}. "
            "ingest에서 dtype=str로 읽어라(foundation.md 결정 A)."
        )

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise ValueError("빈 문자열은 PNU가 아니다.")
        # 숫자만 허용 → 'str로 새어든 float'(예: '1.13e+18', '...0')을 여기서 차단.
        if not s.isdigit():
            raise ValueError(f"PNU는 숫자로만 구성된다(소수점·지수표기 불가): {raw!r}")
    elif isinstance(raw, numbers.Integral):
        # 파이썬/numpy 정수. 음수는 PNU가 아니다.
        if raw < 0:
            raise ValueError(f"음수는 PNU가 아니다: {raw!r}")
        s = str(int(raw))
    else:
        raise TypeError(f"PNU로 받을 수 없는 타입 {type(raw).__name__}: {raw!r}")

    if len(s) > PNU_LENGTH:
        raise ValueError(f"PNU가 {PNU_LENGTH}자리를 넘는다(len={len(s)}): {raw!r}")

    # 선행 0 패딩: 정수화 과정에서 사라진 앞자리(시도 '11' 등은 안전하지만,
    # 일부 코드 체계는 0으로 시작)를 표준 19자리로 복원한다.
    return s.zfill(PNU_LENGTH)


def is_valid_pnu(pnu: str) -> bool:
    """표준화된 PNU가 형식적으로 유효한지 검사한다 (조용한 누수 차단).

    역할: 깨진 PNU를 조용히 흘려보내지 않게 하는 위생 검사. 19자리·숫자·
    필지구분(1 또는 2) 형식을 본다. (실재하는 필지인지는 여기서 보지 않는다 —
    그건 실제 데이터 조인이 결정한다.)

    의도적으로 str만 받는다: 정상 흐름이라면 normalize_pnu 를 이미 거친
    문자열이 들어와야 한다.
    """
    if not isinstance(pnu, str):
        return False
    if len(pnu) != PNU_LENGTH:
        return False
    if not pnu.isdigit():
        return False
    if pnu[_FILJI_GUBUN] not in _VALID_FILJI_GUBUN:
        return False
    return True


def parse_pnu(pnu) -> dict:
    """PNU를 의미 단위로 분해한다 (표준화 → 검증 → 분해).

    역할: 시군구코드·산여부·본번·부번 등 PNU에 인코딩된 정보를 꺼낸다.
    먼저 normalize_pnu 로 표준화하고, is_valid_pnu 로 검증한 뒤 분해한다
    (깨진 PNU는 분해 전에 막는다).

    ★산(임야) 필지: 분해는 하되 *필터링은 하지 않는다*. 재개발은 사실상
    일반 필지(필지구분=1)만 관심이지만, 그 판단은 호출부의 몫이다
    (foundation.md 결정 3). is_san 플래그만 제공한다.

    반환 예:
      {"pnu": "1129010100...", "법정동코드": "1129010100",
       "시군구코드": "11290", "필지구분": "1", "is_san": False,
       "본번": 10, "부번": 0}
    """
    norm = normalize_pnu(pnu)
    if not is_valid_pnu(norm):
        raise ValueError(f"유효하지 않은 PNU: {pnu!r} → {norm!r}")

    filji = norm[_FILJI_GUBUN]
    return {
        "pnu": norm,
        "법정동코드": norm[_LEGAL_DONG_CODE],
        "시군구코드": norm[_SIGUNGU_CODE],
        "필지구분": filji,
        "is_san": filji == "2",          # 2=산(임야대장)
        "본번": int(norm[_BONBEON]),       # 표시·비교 편의를 위해 int로
        "부번": int(norm[_BUBEON]),
    }


def sigungu_code(pnu) -> str:
    """PNU에서 시군구코드(5자리)만 뽑는다 (자치구 필터의 키).

    역할: districts.yaml 의 sigungu_code 와 직접 맞물리는 키. 자치구
    단위 필터·집계가 모두 이 5자리 위에서 일어난다.
    """
    return normalize_pnu(pnu)[_SIGUNGU_CODE]


def filter_by_districts(pnus: Iterable, codes: Iterable[str]) -> list[str]:
    """PNU들을 자치구 설정으로 거른다 ('4개 구만'이 코드가 아니라 설정에서 옴).

    역할: 학습/추론 자치구 집합(redev.config.*_sigungu_codes())을 받아,
    해당 구에 속한 표준 PNU만 남긴다. 자치구 리스트가 코드에 박히지 않게
    하는 연결 고리(규칙7).

    입력: pnus(원천 PNU iterable, 미표준화여도 됨), codes(5자리 시군구코드 집합)
    출력: 조건을 만족하는 *표준화된* PNU 리스트
    """
    code_set = set(codes)   # 멤버십 검사를 O(1)로
    return [
        norm
        for norm in (normalize_pnu(p) for p in pnus)
        if norm[_SIGUNGU_CODE] in code_set
    ]
