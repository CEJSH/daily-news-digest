from __future__ import annotations


DEFAULT_TOP_MIX_TARGET = {
    "경제": 2,
    "산업": 2,
    "기술": 3,
    "금융": 2,
    "정책": 2,
    "국제": 2,
    "사회": 1,
    "라이프": 1,
    "헬스": 1,
    "환경": 1,
    "에너지": 2,
    "모빌리티": 1,
}

# TOP 믹스 최소/최대 규칙 (TOP_LIMIT에 따라 유연 조정)
DEFAULT_TOP_MIX_MIN = {
    "산업": 1,
    "경제": 1,
    "국제": 1,
}
DEFAULT_TOP_MIX_MAX = {
    "정책": 2,
}
DEFAULT_TOP_SOURCE_MAX_PER_OUTLET = 2
