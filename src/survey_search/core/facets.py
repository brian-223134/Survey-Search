"""S1 — facet 분해. DESIGN §S1.

토픽 문자열 하나로는 dense 이웃 한 덩어리밖에 못 봅니다. LLM에게 토픽을 하위 주제
8~16개로 쪼개게 하고, facet마다 표현이 다른 쿼리 1~3개를 받습니다. 동의어·약어·구식
표기를 섞도록 프롬프트에서 요구합니다 ("RAG" / "retrieval-augmented generation" /
"retrieval augmented LM").

**파이프라인에서 LLM을 쓰는 곳은 여기 하나뿐입니다.** 나머지는 전부 결정적입니다.
결과는 디스크 캐시라 같은 토픽 재실행은 호출 0회이고, 그래서 실험도 결정적입니다.

한계를 명시해 둡니다: 이 단계는 LLM의 **사전 지식**에 의존하므로 모델 컷오프 이후의
신조어는 못 냅니다. 그 구멍을 S3(BM25)와 S8(스노우볼링)이 메우는 구조입니다.

키가 없으면 규칙 기반 fallback 으로 내려가고 **그 사실이 stats 에 남습니다** —
조용히 "토픽 1쿼리"로 되돌아가면 facet 을 켠 실험과 끈 실험이 구분되지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from survey_search.assets import DATA_DIR
from survey_search.types import Facet

log = logging.getLogger(__name__)

#: 프롬프트나 파싱 규칙을 바꾸면 올리세요 — 캐시 키에 들어가서 옛 캐시를 무효화합니다.
PROMPT_VERSION = "v1"

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

PROMPT = """You are helping build a literature survey on the topic below.

Topic: {topic}

Break this topic into {n_facets} distinct sub-topics (facets) that a comprehensive
survey would need separate sections for. For each facet, write 1-3 search queries
that would retrieve papers on it from an arXiv abstract index.

Rules for the queries:
- Vary the surface form across queries for the same facet: spell out acronyms in one
  and abbreviate in another (e.g. "RAG" vs "retrieval-augmented generation" vs
  "retrieval augmented language model").
- Include method names, model names, dataset names and benchmark names where you know
  them. These are the tokens that identify recent work.
- Include older/alternative terminology for the same idea where it exists.
- Keep each query under 20 words. No boolean operators, no quotes.

Return ONLY JSON, no markdown fence:
{{"facets": [{{"name": "<short human-readable name>", "queries": ["<q1>", "<q2>"]}}]}}"""


@dataclass
class FacetConfig:
    n_facets: int = 12
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    cache_dir: Path | None = None
    timeout_s: float = 90.0
    max_retries: int = 2
    max_queries_per_facet: int = 3

    def resolved(self) -> FacetConfig:
        """환경변수에서 빈 값을 채웁니다. `.env` 는 호출부가 미리 로드해야 합니다."""
        return FacetConfig(
            n_facets=self.n_facets,
            model=self.model or os.environ.get("SURVEY_SEARCH_FACET_MODEL", DEFAULT_MODEL),
            api_key=self.api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=(self.base_url or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
                      ).rstrip("/"),
            cache_dir=self.cache_dir or (DATA_DIR / "facet_cache"),
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            max_queries_per_facet=self.max_queries_per_facet,
        )


@dataclass
class FacetStats:
    source: str = ""          # "cache" | "llm" | "fallback"
    n_facets: int = 0
    n_queries: int = 0
    llm_calls: int = 0
    elapsed_s: float = 0.0
    model: str = ""
    warnings: list[str] = field(default_factory=list)


def load_dotenv(path: Path | str = ".env") -> int:
    """의존성 없이 `.env` 를 읽어 환경변수에 넣습니다. **기존 값은 덮지 않습니다.**

    `python-dotenv` 를 안 쓰는 이유: 이 한 가지 용도로 의존성을 늘릴 이유가 없고,
    형제 레포의 `.env` 가 `export KEY=value` 형태도 섞여 있어 어차피 관대한 파서가 필요합니다.
    """
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k and v and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def cache_key(topic: str, cfg: FacetConfig) -> str:
    raw = f"{PROMPT_VERSION}|{cfg.model}|{cfg.n_facets}|{topic.strip().lower()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _parse_facets(text: str, cfg: FacetConfig) -> list[Facet]:
    """LLM 응답에서 facet 을 뽑습니다. 마크다운 펜스와 앞뒤 산문을 견딥니다."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    if not t.startswith("{"):
        s, e = t.find("{"), t.rfind("}")
        if s < 0 or e <= s:
            raise ValueError(f"JSON 을 찾을 수 없음: {text[:200]!r}")
        t = t[s : e + 1]

    data = json.loads(t)
    out: list[Facet] = []
    for item in data.get("facets", []):
        name = str(item.get("name", "")).strip()
        queries = [str(q).strip() for q in item.get("queries", []) if str(q).strip()]
        queries = queries[: cfg.max_queries_per_facet]
        if name and queries:
            out.append(Facet(name=name, queries=tuple(queries)))
    if not out:
        raise ValueError("facet 이 하나도 파싱되지 않음")
    return out


def _call_openrouter(topic: str, cfg: FacetConfig) -> str:
    body = json.dumps({
        "model": cfg.model,
        "messages": [{"role": "user",
                      "content": PROMPT.format(topic=topic, n_facets=cfg.n_facets)}],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        f"{cfg.base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            # OpenRouter 가 권장하는 식별 헤더. 없어도 동작하지만 rate limit 진단에 도움이 됩니다.
            "X-Title": "survey-search",
        },
    )
    last: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
                payload = json.loads(r.read())
            return payload["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as e:
            last = e
            if attempt < cfg.max_retries:
                wait = 2 ** attempt
                log.warning("facet LLM 호출 실패(%s), %ds 후 재시도", e, wait)
                time.sleep(wait)
    raise RuntimeError(f"OpenRouter 호출이 {cfg.max_retries + 1}회 모두 실패: {last}")


def fallback_facets(topic: str) -> list[Facet]:
    """LLM 없이 만드는 쿼리 변형. **약한 대체재입니다.**

    사전 지식이 없으니 하위 주제를 만들어낼 수 없고, 표면형만 바꿉니다:
    원문 / 소문자 / 약어 전개 없는 축약형 / 핵심어만. 이걸로 facet 실험을 대신할 수는
    없고, 키가 없을 때 파이프라인이 죽지 않게 하는 용도입니다.
    """
    t = topic.strip()
    words = re.findall(r"[A-Za-z0-9-]+", t)
    stop = {"for", "of", "the", "a", "an", "in", "on", "and", "with", "to"}
    core = " ".join(w for w in words if w.lower() not in stop)
    acronym = "".join(w[0].upper() for w in words if w.lower() not in stop and w[0].isalpha())

    queries = [t]
    if core and core.lower() != t.lower():
        queries.append(core)
    if len(acronym) >= 2:
        queries.append(f"{acronym} {core}")
    return [Facet(name="(fallback)", queries=tuple(dict.fromkeys(queries)))]


def decompose(
    topic: str,
    *,
    config: FacetConfig | None = None,
    use_cache: bool = True,
) -> tuple[list[Facet], FacetStats]:
    """토픽 → facet 목록. 캐시 → LLM → fallback 순으로 시도합니다."""
    cfg = (config or FacetConfig()).resolved()
    stats = FacetStats(model=cfg.model)
    t0 = time.perf_counter()

    cache_path = cfg.cache_dir / f"{cache_key(topic, cfg)}.json"
    if use_cache and cache_path.exists():
        data = json.loads(cache_path.read_text())
        facets = [Facet(name=f["name"], queries=tuple(f["queries"])) for f in data["facets"]]
        stats.source = "cache"
        stats.n_facets = len(facets)
        stats.n_queries = sum(len(f.queries) for f in facets)
        stats.elapsed_s = time.perf_counter() - t0
        return facets, stats

    if not cfg.api_key:
        facets = fallback_facets(topic)
        stats.source = "fallback"
        stats.warnings.append(
            "OPENROUTER_API_KEY 가 없어 규칙 기반 fallback 사용 — facet 실험의 대체재가 못 됩니다"
        )
    else:
        try:
            raw = _call_openrouter(topic, cfg)
            facets = _parse_facets(raw, cfg)
            stats.source = "llm"
            stats.llm_calls = 1
        except (RuntimeError, ValueError, json.JSONDecodeError) as e:
            facets = fallback_facets(topic)
            stats.source = "fallback"
            stats.warnings.append(f"LLM facet 분해 실패 -> fallback: {e}")
            log.warning("facet 분해 실패, fallback 사용: %s", e)

    stats.n_facets = len(facets)
    stats.n_queries = sum(len(f.queries) for f in facets)
    stats.elapsed_s = time.perf_counter() - t0

    # fallback 은 캐시하지 않습니다 — 키를 넣고 다시 돌리면 제대로 나와야 합니다.
    if use_cache and stats.source == "llm":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"topic": topic, "model": cfg.model, "prompt_version": PROMPT_VERSION,
             "facets": [{"name": f.name, "queries": list(f.queries)} for f in facets]},
            ensure_ascii=False, indent=2))

    return facets, stats
