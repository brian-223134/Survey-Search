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
import threading
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
    #: 호출 1회의 **벽시계** 상한. `timeout_s` 와 별개로 반드시 필요합니다 — 이유는
    #: `_read_with_deadline` 주석 참고. 실측 정상 지연이 39~41초(n=3, deepseek-v4-flash,
    #: n_facets=12)라 3배 여유를 뒀습니다. 더 늘리면 죽은 라우팅을 그만큼 오래 붙듭니다.
    deadline_s: float = 120.0
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
            deadline_s=self.deadline_s,
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


def _parse_facets(text: str | None, cfg: FacetConfig) -> list[Facet]:
    """LLM 응답에서 facet 을 뽑습니다. 마크다운 펜스와 앞뒤 산문을 견딥니다.

    **빈 응답도 여기서 걸러야 합니다.** OpenRouter 는 가끔 `content: null` 을 돌려주는데
    (거부·필터·프로바이더 이상), 이걸 그대로 `.strip()` 하면 `AttributeError` 가 나고
    `decompose` 의 fallback 이 그 예외를 안 잡아서 **호출 전체가 죽습니다.**
    3시간짜리 평가가 LLM 응답 하나 때문에 중단된 적이 실제로 있습니다.
    """
    if not text or not text.strip():
        raise ValueError("LLM 이 빈 응답을 돌려줬습니다 (content=None 또는 공백)")
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    if not t.startswith("{"):
        s = t.find("{")
        if s < 0:
            raise ValueError(f"JSON 을 찾을 수 없음: {text[:200]!r}")
        t = t[s:]

    # **뒤에 뭐가 붙어도 첫 객체만 읽습니다.** `json.loads` 는 문자열 전체가 JSON 하나여야
    # 해서, 모델이 객체를 낸 뒤 설명을 덧붙이거나 객체를 하나 더 내면
    # `Extra data: line 1 column 2928` 로 터집니다. 이건 재시도 대상이 아니라 곧장
    # fallback 으로 가는 경로라, 그 토픽의 facet 실험이 조용히 무효가 됩니다.
    # 실제로 170토픽 중 2개가 이걸로 날아갔습니다.
    try:
        data, _ = json.JSONDecoder().raw_decode(t)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패({e}): {text[:200]!r}") from e
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


def _read_with_deadline(req: urllib.request.Request, cfg: FacetConfig) -> dict:
    """요청 1회를 **벽시계 상한** 안에서 끝냅니다.

    `urlopen(timeout=)` 은 소켓 연산 1회의 상한이지 호출 전체의 상한이 아닙니다.
    OpenRouter(Cloudflare 뒤)는 생성이 긴 요청에 **10초마다 912바이트의 keep-alive
    를 흘려보냅니다.** 그때마다 읽기가 성공하니 90초 타이머가 매번 초기화되고,
    상한은 영영 발동하지 않습니다.

    실측: 170토픽 평가가 이 상태로 CPU 0%, 정확히 10초 간격 912바이트 수신만 하며
    16분 넘게 멈춰 있었습니다. 소켓은 살아 있어서 예외도 안 났습니다.

    그래서 데몬 스레드에 실어 join 으로 상한을 겁니다. 상한을 넘긴 스레드는 버리는데,
    소켓 하나가 새는 대신 배치가 무한정 서지 않습니다 — 그 편이 낫습니다.
    """
    box: dict[str, object] = {}

    def work() -> None:
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
                box["ok"] = json.loads(r.read())
        except BaseException as e:  # noqa: BLE001  버린 스레드의 예외를 옮겨 싣습니다
            box["err"] = e

    th = threading.Thread(target=work, daemon=True, name="facet-llm")
    th.start()
    th.join(cfg.deadline_s)
    if th.is_alive():
        raise TimeoutError(
            f"OpenRouter 응답이 {cfg.deadline_s:.0f}초 안에 안 끝났습니다 "
            "(keep-alive 때문에 소켓 타임아웃이 안 걸리는 경우)")
    if "err" in box:
        raise box["err"]  # type: ignore[misc]
    return box["ok"]  # type: ignore[return-value]


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
            payload = _read_with_deadline(req, cfg)
            content = payload["choices"][0]["message"]["content"]
            if not content:
                # 빈 응답은 재시도할 가치가 있습니다 — 프로바이더 일시 장애일 수 있습니다.
                raise ValueError(f"content 가 비어 있음: {str(payload)[:200]}")
            return content
        except (urllib.error.URLError, KeyError, IndexError, TypeError, ValueError,
                json.JSONDecodeError, TimeoutError) as e:
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
        except Exception as e:  # noqa: BLE001
            # **넓게 잡는 것이 의도입니다.** facet 분해는 파이프라인의 편의 기능이지
            # 필수가 아닙니다. 여기서 올라간 예외 하나가 배치 실행 전체를 죽이면
            # 안 됩니다 — 실제로 AttributeError 하나에 3시간 평가가 중단됐습니다.
            # 대신 실패 사실과 사유를 반드시 남깁니다.
            facets = fallback_facets(topic)
            stats.source = "fallback"
            stats.warnings.append(f"LLM facet 분해 실패 -> fallback: {type(e).__name__}: {e}")
            log.warning("facet 분해 실패, fallback 사용: %s", e)

    stats.n_facets = len(facets)
    stats.n_queries = sum(len(f.queries) for f in facets)
    stats.elapsed_s = time.perf_counter() - t0

    # fallback 은 캐시하지 않습니다 — 키를 넣고 다시 돌리면 제대로 나와야 합니다.
    if use_cache and stats.source == "llm":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(
            {"topic": topic, "model": cfg.model, "prompt_version": PROMPT_VERSION,
             "facets": [{"name": f.name, "queries": list(f.queries)} for f in facets]},
            ensure_ascii=False, indent=2)
        # 원자적으로 갈아끼웁니다. 캐시를 채우는 프로세스를 여러 개 띄우면 같은 토픽이
        # 겹칠 수 있고, 그때 write_text 는 반쯤 쓰인 JSON 을 남길 수 있습니다 —
        # 그 파일은 다음 실행에서 조용히 JSONDecodeError 로 터집니다.
        tmp = cache_path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(blob)
        os.replace(tmp, cache_path)

    return facets, stats
