from __future__ import annotations

import html
import re
from typing import Any

_WS_RE = re.compile(r"\s+")  # 공백 정리 시 연속 공백을 단일 공백으로 축약
_TRAILING_TAG_RE = re.compile(r"\s*[\[\(][^\]\)]+[\]\)]\s*$")  # 제목 끝의 괄호/대괄호 태그 제거용
_SOURCE_SEPARATOR_RE = re.compile(r"\s*\|\s*|\s+[–—-]\s+|\s*[·•:｜ㅣ]\s*")  # 제목에서 소스/섹션 구분자 분리용
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # 바이너리/제어문자 검출용
_PNG_SIGNS = ("PNG", "IHDR", "IDAT", "IEND")  # 깨진 텍스트에 섞이는 PNG 시그니처 검출용

_MEDIA_SUFFIXES = ("일보", "신문", "뉴스", "방송", "미디어", "TV", "tv")  # 언론사/미디어명 추정용 접미사

_SECTION_TOKENS = {
    "국제", "정치", "경제", "사회", "산업", "증권", "금융", "IT", "테크", "세계", "글로벌"
}  # 섹션명으로 보이는 토큰 (제목 꼬리 제거 판단에 사용)

_LATIN_MEDIA_TOKENS = {
    "reuters", "bloomberg", "ft", "wsj", "journal", "times", "news", "press", "media"
}  # 영문 매체명/토큰 추정용

_KOREAN_PARTICLE_SUFFIXES = (
    "에게서", "에서", "에게", "까지", "부터", "으로", "로", "과", "와", "을", "를", "은", "는",
    "이", "가", "의", "에", "도", "만"
)  # 중복 제거용 토큰 정규화 시 조사 제거

_KOREAN_VERB_ENDINGS = (
    "했습니다", "하였다", "했다", "한다", "합니다", "하며", "된다", "됐다", "되고", "되며",
    "됩니다", "되는", "했다고", "한다고", "했다며", "한다며", "했다는", "한다는", "이었다",
    "이라며", "이라고", "이다", "였다", "였던", "했던", "했고", "되었다", "되었습니다"
)  # 중복 제거용 토큰 정규화 시 동사 어미 제거

_BRIEFING_HINTS = (
    "브리핑", "모닝 브리핑", "시장 브리핑", "증시 브리핑", "마감 브리핑", "시장 요약", "뉴스 요약",
    "헤드라인", "오늘의 뉴스", "오늘의 주요", "주요 뉴스", "데일리", "daily brief", "market wrap",
    "모닝", "정오", "마감", "아침 회의", "뉴스 정리",
    "오늘의 증시", "증시 요약", "시장 마감", "장 마감", "시황", "시황 요약", "시장 동향",
    "경제 브리핑", "경제 요약", "금융 요약", "주요 경제", "주요 이슈", "이슈 브리핑",
    "morning briefing", "evening briefing", "closing bell", "market close", "market summary",
    "daily briefing", "news briefing", "top headlines", "morning wrap", "evening wrap",
)

_MULTI_TOPIC_JOINERS = (
    "그리고", "또한", "한편", "등", "·", "/", "및", "와", "과", "더불어", "이어", "이에", "반면",
)

def clean_text(s: str) -> str:
    if not s:
        return ""
    # 1) &nbsp; 같은 HTML 엔티티를 문자로 변환
    s = html.unescape(s)

    # 2) NBSP(유니코드) -> 일반 스페이스로
    s = s.replace("\u00a0", " ")

    # 3) 혹시 섞여 들어온 HTML 태그 제거
    s = re.sub(r"<[^>]+>", "", s)

    # 4) 공백 정리
    s = _WS_RE.sub(" ", s).strip()
    return s

def clean_text_ws(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())

def contains_binary(text: str) -> bool:
    if not text:
        return False
    if _CONTROL_RE.search(text):
        return True
    if text.count("�") / max(1, len(text)) > 0.01:
        return True
    upper = text.upper()
    return any(sign in upper for sign in _PNG_SIGNS)

def sanitize_text(text: str) -> str:
    """모델 입력 전에 바이너리/깨진 텍스트를 정화."""
    if not text:
        return ""
    text = _CONTROL_RE.sub(" ", text)
    if text.count("�") / max(1, len(text)) > 0.01:
        return ""
    upper = text.upper()
    if any(sign in upper for sign in _PNG_SIGNS):
        return ""
    return _WS_RE.sub(" ", text).strip()

def _looks_like_source_segment(segment: str) -> bool:
    seg = (segment or "").strip()
    if not seg:
        return False
    seg_lower = seg.lower()
    if seg_lower in _LATIN_MEDIA_TOKENS:
        return True
    if seg in _SECTION_TOKENS:
        return True
    if any(seg.endswith(suf) for suf in _MEDIA_SUFFIXES):
        return True
    return False

def trim_title_noise(title: str, source_name: str | None = None) -> str:
    if not title:
        return ""

    # 1) 제목 끝에 붙는 [신문사 이름] 또는 (신문사 이름) 패턴 제거 (선택사항)
    # title = re.sub(r"[\s\[\(]+[가-힣\w\s]+[\]\)]\s*$", "", title)

    # 2) ' | ', ' - ', ' – ', ' — ' 등 구분자 이후 신문사 이름 제거
    # \s+[\|\-–—]\s+패턴을 뒤에서부터 찾아 제거
    # 예: "제목 - 신문사" -> "제목"
    # 예: "제목 | 신문사" -> "제목"
    
    # 여러 구분자를 지원하는 정규식 (앞뒤 공백 포함)
    pattern = r"\s+[\|\-–—]\s+[^\|\-–—]+$"
    title = re.sub(pattern, "", title)

    # 혹시 한 번에 다 안 지워지는 경우(중첩)를 위해 한 번 더 시도하거나
    # split으로 안전하게 처리
    title = title.split(" | ")[0]
    if " - " in title:
        title = title.rsplit(" - ", 1)[0]

    # 3) 후미 괄호/대괄호 태그 제거 (반복)
    while True:
        new_title = re.sub(_TRAILING_TAG_RE, "", title)
        if new_title == title:
            break
        title = new_title

    # 4) 소스명 제거 (제목 끝 또는 구분자 뒤)
    if source_name:
        src = re.escape(source_name.strip())
        title = re.sub(rf"(?:\s*[\|\-–—·•:｜ㅣ]\s*)?{src}\s*$", "", title, flags=re.IGNORECASE)

    # 5) 구분자 기반 후미 세그먼트 제거 (소스/섹션 추정)
    parts = _SOURCE_SEPARATOR_RE.split(title)
    if len(parts) >= 2:
        tail = parts[-1].strip()
        if _looks_like_source_segment(tail):
            title = re.sub(rf"{re.escape(tail)}\s*$", "", title).strip()
            title = re.sub(r"\s*[\|\-–—·•:｜ㅣ]\s*$", "", title).strip()

    return title.strip()

def get_source_name(entry: Any) -> str:
    """Google News RSS에서 언론사 이름(source.title)을 가져옴."""
    try:
        if hasattr(entry, "source") and hasattr(entry.source, "title"):
            return entry.source.title.strip()
    except Exception:
        pass
    return ""

def normalize_token_for_dedupe(token: str, stopwords: set[str]) -> str:
    tok = (token or "").strip().lower()
    if not tok:
        return ""
    if re.search(r"[가-힣]", tok):
        for suf in _KOREAN_VERB_ENDINGS:
            if tok.endswith(suf) and len(tok) - len(suf) >= 2:
                tok = tok[: -len(suf)]
                break
        for suf in _KOREAN_PARTICLE_SUFFIXES:
            if tok.endswith(suf) and len(tok) - len(suf) >= 2:
                tok = tok[: -len(suf)]
                break
    if not tok or tok in stopwords:
        return ""
    if re.search(r"[가-힣]", tok):
        if len(tok) < 2:
            return ""
    else:
        if len(tok) < 3:
            return ""
    return tok

def split_summary_to_3lines(summary: str) -> list[str]:
    """요약 문자열을 최대 3줄 배열로 변환. (MVP UI용)"""
    s = (summary or "").strip()
    if not s:
        return []

    # 문장 단위 분리(영문/국문 공통) → 최대 3개
    parts = [
        p.strip()
        for p in re.split(r"(?<=[\.\!\?。])\s+|(?<=다\.)\s+", s)
        if p.strip()
    ]
    if len(parts) >= 3:
        return parts[:3]

    # 문장 분리가 애매하면 길이로 균등 분할
    if len(parts) <= 1 and len(s) > 120:
        step = max(40, len(s)//3)
        chunks = [s[i:i+step].strip() for i in range(0, len(s), step)]
        return chunks[:3]

    return parts

def split_summary_to_lines(summary: str, max_lines: int = 3) -> list[str]:
    """문장 경계 기준으로만 분리 (강제 분할 없음)."""
    s = (summary or "").strip()
    if not s:
        return []
    parts = [
        p.strip()
        for p in re.split(r"(?<=[\.\!\?。])\s+|(?<=다\.)\s+", s)
        if p.strip()
    ]
    if parts:
        return parts[:max_lines]
    return [s]

def ensure_three_lines(lines: list[str], fallback_text: str) -> list[str]:
    """요약 라인이 3줄이 되도록 보정한다."""
    cleaned = [clean_text(x) for x in (lines or []) if clean_text(x)]
    if len(cleaned) >= 3:
        return cleaned[:3]

    fallback = clean_text(fallback_text or "")
    if fallback:
        parts = split_summary_to_3lines(fallback)
        if not (len(parts) == 1 and not cleaned):
            for line in parts:
                if line and line not in cleaned:
                    cleaned.append(line)
                if len(cleaned) >= 3:
                    return cleaned[:3]

        if len(cleaned) < 3:
            step = max(20, (len(fallback) + 2) // 3)
            chunks = [fallback[i:i + step].strip() for i in range(0, len(fallback), step)]
            for c in chunks:
                if c and c not in cleaned:
                    cleaned.append(c)
                if len(cleaned) >= 3:
                    return cleaned[:3]

    return cleaned[:3]

def ensure_lines_1_to_3(lines: list[str], fallback_text: str) -> list[str]:
    """요약 라인을 1~3줄로 보정한다. (강제 분할 없음)"""
    if not isinstance(lines, list):
        lines = [str(lines)]
    cleaned = []
    for x in (lines or []):
        base = clean_text(x)
        base = strip_summary_boilerplate(base)
        base = clean_text(base)
        if base:
            cleaned.append(base)
    if cleaned:
        return _dedupe_lines(cleaned)[:3]

    fallback = clean_text(fallback_text or "")
    if not fallback:
        return []
    fallback_lines = split_summary_to_lines(strip_summary_boilerplate(fallback), max_lines=3)
    return _dedupe_lines(fallback_lines[:3] if fallback_lines else [fallback])[:3]

def estimate_read_time_seconds(text: str) -> int:
    """한국어 평균 읽기 속도 ~500자/분 가정. 10초 단위 반올림, 10~40초로 클램프."""
    n = len((text or "").strip())
    if n <= 0:
        return 10
    seconds = (n / 500) * 60
    # 10초 단위 반올림
    rounded = int(round(seconds / 10) * 10)
    return max(10, min(40, rounded))

def normalize_title_for_dedupe(title: str, stopwords: set[str]) -> set[str]:
    t = trim_title_noise(clean_text(title)).lower()
    t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
    toks = [normalize_token_for_dedupe(x, stopwords) for x in t.split()]
    return {x for x in toks if x}

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def strip_summary_boilerplate(text: str) -> str:
    """요약 후보에서 저작권/연락처/주소 등 고정 문구를 제거."""
    if not text:
        return ""
    lines = [line.strip() for line in re.split(r"[\\r\\n]+", text) if line.strip()]
    patterns = [
        r"제호",
        r"대표전화",
        r"주소\\s*:\\s*",
        r"등록번호",
        r"등록일",
        r"발행인",
        r"편집인",
        r"기사배열책임자",
        r"청소년보호책임자",
        r"Copyright",
        r"All\\s*Rights",
        r"Rights\\s*Reserved",
        r"Rights\\s*R",
        r"ⓒ",
        r"무단전재",
        r"재배포",
        r"AI 학습 이용",
        r"열린보도원칙",
        r"반론",
        r"정정 보도",
    ]
    cleaned: list[str] = []
    for line in lines:
        earliest = None
        for pat in patterns:
            m = re.search(pat, line, flags=re.IGNORECASE)
            if m:
                earliest = m.start() if earliest is None else min(earliest, m.start())
        if earliest is not None:
            prefix = line[:earliest].strip()
            if len(prefix) < 12:
                continue
            line = prefix
        line = re.sub(r"[\\s\\-–—·•:｜ㅣ]+$", "", line).strip()
        if not line:
            continue
        cleaned.append(line)
    return clean_text(" ".join(cleaned))

def _dedupe_lines(lines: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for line in lines:
        norm = clean_text(line)
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out

def is_briefing_title_or_text(title: str, summary_text: str) -> bool:
    combined = f"{title} {summary_text}".lower()
    return any(hint.lower() in combined for hint in _BRIEFING_HINTS)

def _topic_tokens(text: str) -> set[str]:
    t = clean_text(text).lower()
    t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
    parts = [normalize_token_for_dedupe(x, stopwords=set()) for x in t.split()]
    return {p for p in parts if p}

def is_multi_topic_summary(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    tokens_list = [_topic_tokens(line) for line in lines if line]
    tokens_list = [t for t in tokens_list if len(t) >= 2]
    if len(tokens_list) < 2:
        return False
    overlaps = []
    for i in range(len(tokens_list) - 1):
        overlaps.append(jaccard(tokens_list[i], tokens_list[i + 1]))
    if overlaps and min(overlaps) < 0.2:
        return True
    joined = " ".join(lines)
    if sum(1 for j in _MULTI_TOPIC_JOINERS if j in joined) >= 1:
        return True
    return False


def jaccard_tokens(a: str, b: str) -> float:
    toks_a = set(clean_text_ws(a).split())
    toks_b = set(clean_text_ws(b).split())
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)
