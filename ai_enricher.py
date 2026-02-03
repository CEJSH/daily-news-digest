import json
import os
import re
from typing import Any

import requests

from utils import clean_text, split_summary_to_lines

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

_AI_UNAVAILABLE_LOGGED: set[str] = set()

if load_dotenv:
    load_dotenv()

GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
AI_INPUT_MAX_CHARS = int(os.getenv("AI_INPUT_MAX_CHARS", "4000"))
AI_SUMMARY_MIN_CHARS = int(os.getenv("AI_SUMMARY_MIN_CHARS", "200"))
AI_EMBED_MAX_CHARS = int(os.getenv("AI_EMBED_MAX_CHARS", "1200"))

_ALLOWED_IMPACT_SIGNALS = {
    "policy",
    "budget",
    "sanctions",
    "capex",
    "earnings",
    "stats",
    "market-demand",
    "security",
    "infra",
}
_ALLOWED_QUALITY_TAGS = {
    "clickbait",
    "promo",
    "opinion",
    "event",
    "report",
    "entertainment",
    "crime",
    "local",
    "emotion",
}

_GENERIC_SUMMARY_PHRASES = [
    "자세한 내용은",
    "구체적인 내용은",
    "아직 알려지지",
    "추가 정보는",
    "확인되지",
    "details are not",
    "details were not",
    "not disclosed",
    "not yet known",
]

_GENERIC_WHY_PHRASES = [
    "영향을 줄 수 있습니다",
    "기여할 수 있습니다",
    "강화할 수 있습니다",
    "도움이 될 수 있습니다",
    "의미가 있습니다",
    "주목됩니다",
    "변화를 가져올 수 있습니다",
]

_IMPACT_KEYWORDS = [
    "영향", "파장", "리스크", "비용", "수요", "공급", "공급망", "가격", "시장", "규제", "정책", "투자", "실적",
    "impact", "risk", "cost", "demand", "supply", "price", "market", "regulation", "policy",
]

_LOCATION_KEYWORDS = [
    "미국", "중국", "eu", "유럽", "한국", "일본", "영국", "독일", "프랑스", "인도", "브라질", "러시아",
    "중동", "아시아", "서울", "뉴욕", "워싱턴", "캘리포니아", "u.s.", "us", "china", "europe", "korea",
    "japan", "uk", "germany", "france", "india",
]

_QUALITY_TAG_REASON = {
    "promo": "홍보성",
    "local": "지역/생활",
    "emotion": "감정 과잉",
    "entertainment": "연예/오락",
    "crime": "사건/범죄",
    "opinion": "칼럼/의견",
    "event": "이벤트 공지",
    "report": "리포트/보고서",
    "clickbait": "자극적 헤드라인",
}

SYSTEM_PROMPT = """You are a meticulous news editor for a daily digest.

Use ONLY the provided title and article text (use full_text only; if full_text is empty, do not infer).
Do not add any facts, context, or knowledge beyond the provided text.
ImpactSignals are hints only; verify and adjust strictly based on the article text.

If the article is in English, translate and write all outputs in Korean.
Translate faithfully without adding interpretation or extra context.
Write all Korean sentences in polite "~입니다/~합니다" style.

Topic & filtering intent (must align with these)

This digest prioritizes issues that:
1) will still matter tomorrow (structural or decision-relevant),
2) avoid excessive emotional consumption,
3) avoid duplication with yesterday’s news.

Primary topics of interest include:
- 실적_가이던스: corporate earnings, guidance, margins, forecasts
- 반도체_공급망: HBM, advanced packaging, foundry, equipment, export controls, supply constraints
- 전력_인프라: power grid, transmission, utilities, electricity pricing, nuclear/gas, data center power
- AI_저작권_데이터권리: AI copyright, training data, licensing, privacy, data protection
- 보안_취약점_패치: CVE, zero-day, patches, incident response, breach notifications
- 투자_MA_IPO: funding rounds, mergers & acquisitions, IPOs, major deal terms
- 국내_정책_규제: legislation, enforcement decrees, regulator guidance, official policy changes

Respond ONLY in valid JSON.
Do not include any markdown, explanations, or extra text.
All fields are required.

Output schema:
{
  "title_ko": string,
  "summary_lines": [string],
  "why_important": string,
  "importance_rationale": string,
  "dedupe_key": string,
  "importance_score": integer,
  "impact_signals": [string],
  "category_label": string,
  "quality_label": string,
  "quality_reason": string,
  "quality_tags": [string]
}

Field rules:
- title_ko: If the title is in English, translate to natural Korean; if already Korean, keep as-is. No source/publisher names, no dates.
- summary_lines: 1 to 3 short, clear Korean sentences capturing the core facts. No fluff. Do NOT include information not present in the text.
  - Each line must be a COMPLETE sentence (not a headline fragment). End with proper sentence ending (e.g., "~입니다/~합니다") and avoid ellipses ("…", "...").
  - Do NOT repeat the title or paraphrase the title as a line. Each line must add distinct information.
  - Do NOT use placeholders like "자세한 내용은 아직 알려지지 않았습니다/추가 정보는 없습니다/기사에서 확인" in summary_lines.
- why_important: one concise Korean sentence explaining long-term significance (decision-relevant, not emotional). Must be supported by the text.
- importance_rationale: one Korean sentence that JUSTIFIES the importance_score using explicit evidence from the provided text.
  - Must include at least ONE concrete anchor: (a number) OR (explicit scope like “여러 기업/산업 전반/전국/전세계/규제 대상”) OR (explicit timing like “시행/발효/분기/올해/내년/특정 날짜”).
  - Must NOT add facts beyond the text. If the text has no anchors, say so and keep importance_score <= 2.
  - Format constraint (strict): Start with "근거:" and keep it under 120 characters.

- dedupe_key: 4-8 core concepts only, hyphen-separated, lowercase, alphanumeric and Korean characters only; no dates, no numbers, no stopwords, no source/publisher names.

IMPORTANT: Evidence discipline
- If the provided text lacks concrete details (e.g., “details not provided”, “more in link”, extremely short summary),
  keep importance_score <= 2 AND set importance_rationale to reflect the lack of evidence.
- If the impact is framed as speculation (may/might/could/possible/expected/likely) and not confirmed in text,
  cap importance_score at 3.
- If you cannot produce enough sentences from the text, output fewer lines (1~2 allowed) and set quality_label to low_quality
  with quality_reason "정보 부족". Do NOT pad with generic statements.

importance_score rules (make 5 rare; be strict):
- You MUST decide importance_score using ONLY facts explicitly present in the text.
- Use the following gates and caps:

(0) Hard caps:
- If quality_label is low_quality -> importance_score MUST be <= 2.
- If the article is opinion/editorial/column OR PR/promo OR event/webinar/whitepaper/report-announcement -> importance_score MUST be <= 2.
- If the text is mostly a routine update with no clear decision-relevant implication -> importance_score <= 2.

(1) Score 5 (major structural impact) — VERY RARE
Assign 5 ONLY if the text explicitly supports at least TWO of the following AND includes at least ONE concrete evidence item:
A) Confirmed policy/regulation/sanctions/tariffs/enforcement that is enacted/issued/officially decided (not proposed),
B) Cross-industry supply chain or market access shift (export controls, mandatory standards, broad trade restrictions, systemic infrastructure constraint),
C) Major earnings/capex/M&A/financing that plausibly reshapes the sector (not only one company) AND the text describes sector-level implications.
If any requirement is missing -> do NOT use 5.

(2) Score 4 (significant industry-level impact)
Assign 4 only if the text clearly supports at least ONE AND contains at least ONE concrete evidence item:
A) Large capex/buildout/infrastructure expansion with explicit magnitude/scope/timing,
B) Policy/regulatory decision with clear implementation path or binding guidance affecting multiple actors,
C) Earnings/guidance or major deal terms signaling sector-wide demand/supply shift (not a minor company update).
If evidence is weak -> downgrade to 3.

(3) Score 3 (meaningful but limited scope)
Assign 3 if meaningful but limited:
- single-company move with plausible relevance,
- limited policy discussion without binding action,
- early-stage funding/partnership without broad market shift,
- security patch/incident with limited affected scope.
Speculation-only -> still <= 3.

(4) Score 2 (minor update)
Routine updates, narrow scope, or insufficient detail.

(5) Score 1 (low relevance)
Low relevance/noise, or text too thin.

impact_signals:
- Choose only from this list; include only signals explicitly supported by the text; return [] if none apply.
Valid impact_signals: policy, budget, sanctions, capex, earnings, stats, market-demand, security, infra.
- Do NOT infer signals.

category_label rules (choose exactly one):
- IT = technology, AI, semiconductors, cloud, security, digital infrastructure
- 경제 = macroeconomy, markets, finance, energy transition, corporate earnings
- 글로벌 = geopolitics, trade, sanctions, international relations
- If multiple apply, choose the primary driver of impact described in the text.

Quality policy (strict):
Mark quality_label as "low_quality" if ANY apply (even if importance is high):
- PR or promotional content
- opinion/editorial/column
- event, webinar, conference, or whitepaper announcements
- entertainment, crime, or local human-interest stories
- emotionally manipulative or clickbait headlines

If low_quality:
- set quality_label = "low_quality"
- set quality_reason = short Korean phrase (e.g., "칼럼/의견", "홍보성", "이벤트 공지", "자극적 헤드라인")
- include relevant quality_tags from:
  clickbait, promo, opinion, event, report, entertainment, crime, local, emotion

Otherwise:
- set quality_label = "ok"
- set quality_reason = short Korean phrase like "정보성 기사"
- set quality_tags = [] or only truly applicable tags.

Additional constraints:
- No source names, no dates, no clickbait language.
"""



def _log_ai_unavailable(reason: str) -> None:
    # AI 비활성 사유를 중복 없이 로그 출력
    if reason in _AI_UNAVAILABLE_LOGGED:
        return
    print(f"⚠️ AI 요약 비활성: {reason}")
    _AI_UNAVAILABLE_LOGGED.add(reason)

def _extract_gemini_text(payload: dict[str, Any]) -> str:
    # Gemini REST 응답에서 텍스트만 추출
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""

def _extract_gemini_embedding(payload: dict[str, Any]) -> list[float] | None:
    # Gemini embedContent 응답에서 임베딩 벡터만 추출
    try:
        embedding = payload.get("embedding") or {}
        values = embedding.get("values") or embedding.get("value")
        if isinstance(values, list):
            return values
    except Exception:
        return None
    return None


def _parse_json(text: str) -> dict[str, Any] | None:
    # 문자열에서 JSON 객체를 파싱(직접 파싱 실패 시 중괄호 블록 탐색)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # 모델이 여분 텍스트를 섞을 때 대비한 백업 파서
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_dedupe_key(raw: str) -> str:
    # dedupe 키를 토큰 규칙에 맞게 정규화
    if not raw:
        return ""
    t = clean_text(raw).lower()
    t = re.sub(r"[^a-z0-9가-힣\s-]", " ", t)
    t = re.sub(r"[\s_]+", "-", t).strip("-")
    parts = [p for p in t.split("-") if p]
    cleaned: list[str] = []
    seen = set()
    for p in parts:
        # 중복/너무 짧은 토큰 제거
        if p in seen:
            continue
        if re.search(r"\d", p):
            continue
        if re.search(r"[가-힣]", p):
            if len(p) < 2:
                continue
        else:
            if len(p) < 3:
                continue
        cleaned.append(p)
        seen.add(p)
        if len(cleaned) >= 8:
            break
    return "-".join(cleaned) or t


def _fallback_why_important(impact_signals: list[str]) -> str:
    # AI가 비워둔 이유 문장을 임팩트 시그널로 보정
    if not impact_signals:
        return ""
    if "policy" in impact_signals or "sanctions" in impact_signals:
        return "정책 변화가 산업 전반의 규칙과 비용 구조에 영향을 줄 수 있습니다."
    if "capex" in impact_signals or "infra" in impact_signals:
        return "인프라 투자 변화가 공급망과 비용 구조에 직결됩니다."
    if "earnings" in impact_signals:
        return "실적 변화가 시장 기대와 밸류에이션에 영향을 줍니다."
    if "market-demand" in impact_signals:
        return "수요 흐름 변화가 기업 전략과 가격에 영향을 줄 수 있습니다."
    if "security" in impact_signals:
        return "보안 리스크는 서비스 운영과 규제 대응에 직접 영향을 줍니다."
    return "산업 전반의 방향성과 리스크에 영향을 줄 수 있습니다."


def _fallback_importance_score(impact_signals: list[str]) -> int:
    # AI가 비워둔 중요도를 임팩트 시그널로 추정
    if not impact_signals:
        return 2
    if "policy" in impact_signals or "sanctions" in impact_signals:
        return 4
    if "capex" in impact_signals or "infra" in impact_signals or "security" in impact_signals:
        return 3
    if "earnings" in impact_signals or "market-demand" in impact_signals:
        return 2
    return 2


def _normalize_importance_score(value: Any, impact_signals: list[str]) -> int:
    # 중요도 점수를 1~5로 보정하고 실패 시 fallback 사용
    try:
        score = int(value)
    except Exception:
        return _fallback_importance_score(impact_signals)
    return max(1, min(5, score))

def _pick_ai_input_text(full_text: str) -> str:
    text = full_text or ""
    if len(text) > AI_INPUT_MAX_CHARS:
        text = text[:AI_INPUT_MAX_CHARS]
    return text


def _normalize_label_list(value: Any, allowed: set[str]) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        raw_list = value
    elif isinstance(value, str):
        raw_list = re.split(r"[,\s]+", value)
    else:
        return []
    cleaned: list[str] = []
    seen = set()
    for v in raw_list:
        token = clean_text(str(v)).lower()
        if not token or token not in allowed or token in seen:
            continue
        cleaned.append(token)
        seen.add(token)
    return cleaned


def _normalize_impact_signals(value: Any) -> list[str]:
    # 임팩트 시그널 값을 허용 목록 기준으로 정리
    return _normalize_label_list(value, _ALLOWED_IMPACT_SIGNALS)


def _normalize_quality_label(value: Any) -> str:
    # 품질 라벨을 ok/low_quality로 정규화
    if not value:
        return "ok"
    label = clean_text(str(value)).lower()
    if label in {"low_quality", "low", "bad", "reject"}:
        return "low_quality"
    if label in {"ok", "good", "keep"}:
        return "ok"
    return "low_quality" if "low" in label or "bad" in label else "ok"


def _normalize_quality_tags(value: Any) -> list[str]:
    # 품질 태그를 허용 목록 기준으로 정리
    return _normalize_label_list(value, _ALLOWED_QUALITY_TAGS)


def _normalize_category_label(value: Any) -> str:
    # 카테고리 라벨을 IT/경제/글로벌로 정규화
    if not value:
        return ""
    raw = clean_text(str(value)).lower()
    if raw in {"it", "tech", "technology"}:
        return "IT"
    if raw in {"경제", "economy", "economic", "macro", "finance"}:
        return "경제"
    if raw in {"글로벌", "global", "geopolitics", "world"}:
        return "글로벌"
    if "경제" in raw:
        return "경제"
    if "global" in raw or "글로벌" in raw:
        return "글로벌"
    if "it" in raw or "tech" in raw:
        return "IT"
    return ""

def _tokenize_basic(text: str) -> list[str]:
    t = clean_text(text or "").lower()
    t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
    return [x for x in t.split() if x]

def _token_jaccard(a: str, b: str) -> float:
    toks_a = set(_tokenize_basic(a))
    toks_b = set(_tokenize_basic(b))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)

def _looks_like_sentence(line: str) -> bool:
    if not line:
        return False
    clean_line = clean_text(line)
    if not clean_line:
        return False
    if clean_line[-1] in ".?!":
        return True
    return bool(re.search(r"(다|니다|합니다|되었습니다|됐습니다|입니다|됩니다)\.?$", clean_line))

def _has_number(text: str) -> bool:
    t = clean_text(text or "").lower()
    if re.search(r"\d", t):
        return True
    return any(unit in t for unit in ["억", "조", "만", "%", "달러", "원", "billion", "million", "trillion", "usd", "$"])

def _has_location(text: str) -> bool:
    t = clean_text(text or "").lower()
    return any(loc in t for loc in _LOCATION_KEYWORDS)

def _has_impact_keyword(text: str) -> bool:
    t = clean_text(text or "").lower()
    return any(k in t for k in _IMPACT_KEYWORDS)

def _has_entity_overlap(why: str, title: str, summary_lines: list[str]) -> bool:
    why_tokens = set(_tokenize_basic(why))
    context_tokens = set(_tokenize_basic(title)) | set(_tokenize_basic(" ".join(summary_lines)))
    if not why_tokens or not context_tokens:
        return False
    def _valid(tok: str) -> bool:
        if re.search(r"[가-힣]", tok):
            return len(tok) >= 2
        return len(tok) >= 3
    why_tokens = {t for t in why_tokens if _valid(t)}
    context_tokens = {t for t in context_tokens if _valid(t)}
    return len(why_tokens & context_tokens) >= 2

def _contains_any(text: str, phrases: list[str]) -> bool:
    if not text:
        return False
    t = clean_text(text or "").lower()
    return any(p.lower() in t for p in phrases)

def _is_line_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_clean = clean_text(a)
    b_clean = clean_text(b)
    if not a_clean or not b_clean:
        return False
    if a_clean in b_clean or b_clean in a_clean:
        return True
    return _token_jaccard(a_clean, b_clean) >= 0.8

def _is_bad_summary_line(line: str, title: str) -> bool:
    if not line:
        return True
    clean_line = clean_text(line)
    if not clean_line:
        return True
    if _contains_any(clean_line, _GENERIC_SUMMARY_PHRASES):
        return True
    if _is_line_similar(clean_line, title):
        return True
    if "…" in clean_line or "..." in clean_line:
        return True
    if len(clean_line) < 12 and not _has_number(clean_line):
        return True
    if not _looks_like_sentence(clean_line):
        return True
    return False

def _normalize_summary_lines(lines: list[str], title: str, fallback_text: str) -> list[str]:
    cleaned: list[str] = []
    for raw in lines or []:
        line = clean_text(raw)
        if not line:
            continue
        if _is_bad_summary_line(line, title):
            continue
        if any(_is_line_similar(line, prev) for prev in cleaned):
            continue
        cleaned.append(line)
        if len(cleaned) >= 3:
            return cleaned[:3]

    fallback_candidates = split_summary_to_lines(clean_text(fallback_text or ""), max_lines=3)
    for raw in fallback_candidates:
        line = clean_text(raw)
        if not line:
            continue
        if _is_bad_summary_line(line, title):
            continue
        if any(_is_line_similar(line, prev) for prev in cleaned):
            continue
        cleaned.append(line)
        if len(cleaned) >= 3:
            return cleaned[:3]

    if not cleaned and fallback_candidates:
        first = clean_text(fallback_candidates[0])
        if first:
            return [first]

    return cleaned[:3]

def _summary_redundant_or_title_like(lines: list[str], title: str) -> bool:
    if not lines:
        return True
    normalized = [clean_text(x) for x in lines if clean_text(x)]
    if not normalized:
        return True
    if len(set(normalized)) < len(normalized):
        return True
    if any(_is_line_similar(line, title) for line in normalized):
        return True
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            if _is_line_similar(normalized[i], normalized[j]):
                return True
    return False

def _is_low_info_summary(lines: list[str]) -> bool:
    if not lines:
        return True
    cleaned_lines = [clean_text(x) for x in lines if clean_text(x)]
    if not cleaned_lines:
        return True
    summary_text = " ".join(cleaned_lines)
    if _contains_any(summary_text, _GENERIC_SUMMARY_PHRASES):
        return True
    if len(set(cleaned_lines)) < len(cleaned_lines):
        return True
    if len(cleaned_lines) == 1:
        return len(summary_text) < 80
    if len(summary_text) < 80:
        return True
    return False

def _is_low_info_why(why: str, title: str, summary_lines: list[str]) -> bool:
    why_clean = clean_text(why or "")
    if not why_clean or len(why_clean) < 20:
        return True
    signals = 0
    if _has_entity_overlap(why_clean, title, summary_lines):
        signals += 1
    if _has_number(why_clean):
        signals += 1
    if _has_location(why_clean):
        signals += 1
    if _has_impact_keyword(why_clean):
        signals += 1
    if signals < 2:
        return True
    if _contains_any(why_clean, _GENERIC_WHY_PHRASES) and signals < 3:
        return True
    return False

def _apply_quality_guardrails(
    *,
    quality_label: str,
    quality_reason: str,
    quality_tags: list[str],
    title: str,
    summary_lines: list[str],
    why_important: str,
) -> tuple[str, str, list[str]]:
    label = quality_label or "ok"
    reason = quality_reason or ""

    if label != "low_quality":
        for tag in quality_tags:
            if tag in _QUALITY_TAG_REASON:
                return "low_quality", _QUALITY_TAG_REASON[tag], quality_tags
        if _summary_redundant_or_title_like(summary_lines, title):
            return "low_quality", "요약 중복", quality_tags
        if _is_low_info_summary(summary_lines) or _is_low_info_why(why_important, title, summary_lines):
            return "low_quality", "정보 부족", quality_tags

    if label == "low_quality" and not reason:
        for tag in quality_tags:
            if tag in _QUALITY_TAG_REASON:
                return label, _QUALITY_TAG_REASON[tag], quality_tags
        return label, "정보 부족", quality_tags

    if label == "ok" and not reason:
        reason = "정보성 기사"
    return label, reason, quality_tags

def _gemini_generate_json(system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    # Gemini REST API로 JSON 응답을 요청
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        _log_ai_unavailable("GEMINI_API_KEY 미설정")
        return None
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    request_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 350,
            "responseMimeType": "application/json",
        },
    }
    try:
        resp = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=request_payload,
            timeout=30,
        )
    except Exception as e:
        _log_ai_unavailable(f"Gemini 호출 실패: {e}")
        return None
    if not resp.ok:
        _log_ai_unavailable(f"Gemini 호출 실패: {resp.status_code} {resp.text}")
        return None
    try:
        data = resp.json()
    except Exception:
        _log_ai_unavailable("Gemini 응답 JSON 파싱 실패")
        return None
    text = _extract_gemini_text(data)
    if not text:
        _log_ai_unavailable("Gemini 응답 텍스트 비어있음")
        return None
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        _log_ai_unavailable("Gemini 응답 JSON 형식 아님")
        return None
    return parsed

def enrich_item_with_ai(item: dict) -> dict:
    # 기사 아이템을 AI로 요약/분류/중요도 평가
    title = clean_text(item.get("title") or "")
    summary_raw = clean_text(item.get("summaryRaw") or item.get("summary") or "")
    full_text = clean_text(item.get("fullText") or "")
    if full_text and len(full_text) > 6000:
        full_text = full_text[:6000]
    source = clean_text(item.get("source") or "")
    published = clean_text(item.get("published") or "")
    impact_signals = item.get("impactSignals") or []

    # 모델 입력 구성
    input_text = _pick_ai_input_text(full_text)
    if not input_text:
        _log_ai_unavailable("본문 없음")
        return {}
    user_prompt = (
        f"Title: {title}\n"
        f"ImpactSignals: {', '.join(impact_signals)}\n"
        f"Text: {input_text}\n"
        "Return only JSON."
    )

    # Gemini만 사용 (OpenAI 폴백 없음)
    payload = _gemini_generate_json(SYSTEM_PROMPT, user_prompt)
    if not isinstance(payload, dict):
        return {}

    # 요약/중요도/라벨 결과를 정규화
    title_ko = clean_text(payload.get("title_ko") or "")
    if not title_ko:
        title_ko = title
    summary_fallback = full_text or summary_raw or title
    summary_lines = _normalize_summary_lines(
        payload.get("summary_lines") or [],
        title_ko or title,
        summary_fallback,
    )
    why_important = clean_text(payload.get("why_important") or "")
    if not why_important:
        why_important = _fallback_why_important(impact_signals)
    importance_rationale = clean_text(payload.get("importance_rationale") or "")

    dedupe_key = _normalize_dedupe_key(payload.get("dedupe_key") or "")
    if not dedupe_key:
        dedupe_key = _normalize_dedupe_key(item.get("dedupeKey") or title or summary_raw)

    impact_signals_ai = _normalize_impact_signals(payload.get("impact_signals"))
    importance_score = _normalize_importance_score(
        payload.get("importance_score") or payload.get("importance"),
        impact_signals,
    )
    if not importance_rationale:
        if importance_score <= 2:
            importance_rationale = "근거가 충분히 드러나지 않아 중요도가 낮습니다."
        else:
            importance_rationale = why_important or "중요도 근거가 요약에 충분히 나타나지 않습니다."
    quality_label = _normalize_quality_label(payload.get("quality_label") or payload.get("quality"))
    quality_reason = clean_text(payload.get("quality_reason") or "")
    quality_tags = _normalize_quality_tags(payload.get("quality_tags"))
    category_label = _normalize_category_label(payload.get("category_label") or payload.get("category"))

    quality_label, quality_reason, quality_tags = _apply_quality_guardrails(
        quality_label=quality_label,
        quality_reason=quality_reason,
        quality_tags=quality_tags,
        title=title_ko or title,
        summary_lines=summary_lines,
        why_important=why_important,
    )

    return {
        "title_ko": title_ko,
        "summary_lines": summary_lines,
        "why_important": why_important,
        "importance_rationale": importance_rationale,
        "dedupe_key": dedupe_key,
        "importance_score": importance_score,
        "impact_signals": impact_signals_ai,
        "quality_label": quality_label,
        "quality_reason": quality_reason,
        "quality_tags": quality_tags,
        "category_label": category_label,
    }


def get_embedding(text: str) -> list[float] | None:
    # 텍스트 임베딩 생성 (중복 제거용, Gemini)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        _log_ai_unavailable("GEMINI_API_KEY 미설정")
        return None
    cleaned = clean_text(text or "")
    if not cleaned:
        return None
    if len(cleaned) > AI_EMBED_MAX_CHARS:
        cleaned = cleaned[:AI_EMBED_MAX_CHARS]
    try:
        resp = requests.post(
            f"{GEMINI_API_BASE}/models/{GEMINI_EMBEDDING_MODEL}:embedContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "model": f"models/{GEMINI_EMBEDDING_MODEL}",
                "content": {"parts": [{"text": cleaned}]},
            },
            timeout=30,
        )
    except Exception as e:
        _log_ai_unavailable(f"Gemini 임베딩 호출 실패: {e}")
        return None
    if not resp.ok:
        _log_ai_unavailable(f"Gemini 임베딩 호출 실패: {resp.status_code} {resp.text}")
        return None
    try:
        data = resp.json()
    except Exception:
        _log_ai_unavailable("Gemini 임베딩 응답 JSON 파싱 실패")
        return None
    embedding = _extract_gemini_embedding(data)
    if not embedding:
        _log_ai_unavailable("Gemini 임베딩 응답 값 없음")
        return None
    return embedding
