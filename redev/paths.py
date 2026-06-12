"""paths — 데이터 루트 단일 입구 (작업 디렉토리 무관).

역할: `_data/...` 경로의 *유일한* 기준점이다. 다른 모듈은 `Path("_data/...")`를
직접 쓰지 않고 여기 `DATA`를 통해 경로를 만든다.

왜 필요한가: 모듈들이 `Path("_data/raw/...")`처럼 **상대경로**를 쓰면 "어디서 실행하느냐"
(현재 작업 디렉토리)에 결과가 묶인다. pipeline 레포 루트에서 돌리면 맞지만, 형제 레포
(backend)가 자기 폴더에서 `redev`를 import해 돌리면 `backend/_data/...`를 찾다 깨진다
(Phase 8 데모 구동 사고). config/__init__.py가 YAML을 `__file__` 기준으로 읽어 CWD를
없앤 것과 같은 원리를, 데이터 경로에도 적용한다.

기준점: 이 파일은 `redev/paths.py` → `parents[1]`이 pipeline 레포 루트. 그 아래 `_data`.
환경변수 `REDEV_DATA_DIR`로 덮어쓸 수 있다(배포·테스트에서 데이터 위치 분리용).
"""

import os
from pathlib import Path

# redev/paths.py 기준: parents[0]=redev/, parents[1]=pipeline 레포 루트.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REDEV_DATA_DIR", REPO_ROOT / "_data"))
