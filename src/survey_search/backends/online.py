"""온라인 백엔드 — arXiv API + Semantic Scholar.

**왜 필요한가.** 로컬 인덱스는 2026-08-04 에서 멈춘 스냅샷입니다. 고정 코퍼스가
원리적으로 못 하는 일이 두 가지 있고, 둘 다 이 프로젝트의 주제와 직결됩니다:

1. **컷오프 이후 논문** — 스냅샷 이후 arXiv 에 올라온 것은 존재하지 않는 것으로 취급됩니다.
   랭킹 축의 최신성(S6)을 아무리 고쳐도 데이터 축의 컷오프는 남습니다.
2. **인용 엣지** — arXiv 메타데이터에는 참고문헌이 없습니다. S2 는 줍니다.
   스노우볼링(S8)이 지금까지 막혀 있던 이유가 이것이고, 인용을 타고 도는 검색은
   dense 유사도도 BM25 도 못 하는 완전히 다른 신호입니다.

**결정성은 캐시로 지킵니다.** 온라인 호출은 비결정적이라 ablation 방법론을 깨뜨립니다.
그래서 모든 응답을 쿼리 키로 디스크에 캐시합니다 — facet 분해와 같은 방식입니다.
두 번째 실행부터는 네트워크 호출 0회이고 결과가 완전히 재현됩니다.

**로컬을 대체하지 않습니다.** 로컬 = recall 과 결정성, 온라인 = 꼬리(컷오프 이후)와
인용 엣지. `HybridBackend` 가 둘을 합칩니다.

API 키: `SEMANTIC_SCHOLAR_API_KEY` (없으면 익명 — 스로틀이 훨씬 셉니다).
arXiv API 는 키가 필요 없지만 **초당 1회** 이상 부르지 말라는 것이 공식 요청사항입니다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from survey_search.assets import DATA_DIR
from survey_search.backends.base import Hit
from survey_search.core.dedup import strip_version
from survey_search.types import Paper

log = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper"

ATOM = "{http://www.w3.org/2005/Atom}"

#: arXiv 공식 요청사항. 지키지 않으면 차단됩니다.
ARXIV_MIN_INTERVAL_S = 3.0


@dataclass
class OnlineStats:
    """무음 폐기 금지 — 네트워크 호출·캐시 히트·실패를 전부 셉니다."""

    arxiv_calls: int = 0
    s2_calls: int = 0
    cache_hits: int = 0
    throttled: int = 0          # 429 를 맞고 재시도한 횟수
    failures: int = 0
    errors: list[str] = field(default_factory=list)


class _Cache:
    """쿼리 키 → JSON 응답. 이게 있어야 실험이 결정적입니다."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str) -> Path:
        h = hashlib.sha1(key.encode()).hexdigest()[:20]
        return self.root / f"{kind}_{h}.json"

    def get(self, kind: str, key: str):
        p = self._path(kind, key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, kind: str, key: str, value) -> None:
        self._path(kind, key).write_text(json.dumps(value, ensure_ascii=False))


def _http_json(req: urllib.request.Request, *, timeout: float, stats: OnlineStats,
               max_try: int = 6, wait: float = 0.5):
    """429 를 재시도 횟수로 세지 않습니다 — SurveyForge `scripts/fetch_citations.py` 가
    실측으로 정착시킨 규칙입니다. 429 는 실패가 아니라 속도 신호입니다."""
    last = None
    tries = 0
    while tries < max_try:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                stats.throttled += 1
                time.sleep(wait)
                continue
            last = f"HTTP {e.code}"
            tries += 1
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = str(e)
            tries += 1
        time.sleep(wait * (2 ** tries))
    stats.failures += 1
    stats.errors.append(str(last))
    return None


class OnlineBackend:
    """arXiv 검색 + S2 인용 그래프. `Backend` 와 `CitationBackend` 를 구현합니다.

    dense 검색은 **하지 않습니다** — arXiv API 는 어휘 검색만 제공합니다.
    `dense_search` 는 `lexical_search` 로 위임하고 그 사실을 로그에 남깁니다.
    임베딩 기반 검색이 필요하면 로컬 백엔드와 함께 `HybridBackend` 로 쓰세요.
    """

    name = "online-arxiv-s2"

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        s2_api_key: str | None = None,
        timeout_s: float = 30.0,
        offline: bool = False,
    ) -> None:
        self.cache = _Cache(cache_dir or (DATA_DIR / "online_cache"))
        self.s2_key = s2_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        self.timeout_s = timeout_s
        #: True 면 캐시에 없는 것은 빈 결과를 냅니다 — 네트워크 없이 실험을 재현할 때
        self.offline = offline
        self.stats = OnlineStats()
        self._last_arxiv_call = 0.0

    # --- arXiv ---------------------------------------------------------------

    def _arxiv_query(self, query: str, top_k: int) -> list[dict]:
        key = f"{query}|{top_k}"
        cached = self.cache.get("arxiv", key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        if self.offline:
            return []

        # arXiv 는 초당 1회 이상 부르지 말라고 명시합니다. 어기면 차단됩니다.
        gap = time.monotonic() - self._last_arxiv_call
        if gap < ARXIV_MIN_INTERVAL_S:
            time.sleep(ARXIV_MIN_INTERVAL_S - gap)

        params = urllib.parse.urlencode({
            "search_query": f'all:"{query}"' if " " in query else f"all:{query}",
            "start": 0,
            "max_results": min(top_k, 200),   # arXiv 권장 상한
            "sortBy": "relevance",
        })
        req = urllib.request.Request(f"{ARXIV_API}?{params}",
                                     headers={"User-Agent": "survey-search/0.1"})
        self.stats.arxiv_calls += 1
        self._last_arxiv_call = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                xml = r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            self.stats.failures += 1
            self.stats.errors.append(f"arxiv: {e}")
            return []

        out = _parse_arxiv_atom(xml)
        self.cache.put("arxiv", key, out)
        return out

    # --- Backend 프로토콜 -------------------------------------------------------

    def dense_search(self, queries: list[str], top_k: int,
                     field: str = "title_abs") -> list[list[Hit]]:
        log.info("OnlineBackend 에는 dense 검색이 없습니다 — 어휘 검색으로 위임합니다")
        return self.lexical_search(queries, top_k)

    def lexical_search(self, queries: list[str], top_k: int) -> list[list[Hit]]:
        out = []
        for q in queries:
            recs = self._arxiv_query(q, top_k)
            # arXiv 는 점수를 주지 않습니다. 순위를 점수로 씁니다 — 어차피 RRF 가
            # 순위만 쓰므로 융합 결과는 동일합니다.
            out.append([(r["paper_id"], 1.0 / (i + 1)) for i, r in enumerate(recs)])
        return out

    def get_papers(self, paper_ids: list[str]) -> list[Paper]:
        """캐시에 있는 것만 돌려줍니다. 없는 id 는 결과에서 빠지고, 호출부가 셀 수 있습니다."""
        known: dict[str, dict] = {}
        for p in self.cache.root.glob("arxiv_*.json"):
            for r in json.loads(p.read_text()):
                known[r["paper_id"]] = r
        out = []
        for pid in paper_ids:
            r = known.get(pid) or known.get(strip_version(pid))
            if r:
                out.append(_to_paper(r))
        return out

    def filter_ids(self, *, date_min=None, date_max=None, categories=None):
        """온라인 백엔드는 전체 id 집합을 모릅니다 — 사전 필터가 불가능합니다.
        `None`(제한 없음)을 돌려주고, 날짜/카테고리 제한은 호출부가 사후에 겁니다."""
        if date_min or date_max or categories:
            log.info("온라인 백엔드는 사전 필터를 못 합니다 — 사후 필터로 처리하세요")
        return None

    # --- CitationBackend (S8 스노우볼링) -----------------------------------------

    def references(self, paper_id: str) -> list[str]:
        """이 논문이 인용한 논문들(후방). arXiv id 를 가진 것만 돌려줍니다."""
        return self._s2_edges(paper_id, "references")

    def cited_by(self, paper_id: str) -> list[str]:
        """이 논문을 인용한 논문들(전방)."""
        return self._s2_edges(paper_id, "citations")

    def _s2_edges(self, paper_id: str, kind: str, limit: int = 500) -> list[str]:
        base = strip_version(paper_id)
        key = f"{base}|{kind}|{limit}"
        cached = self.cache.get("s2", key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        if self.offline:
            return []

        url = f"{S2_PAPER}/arXiv:{base}/{kind}?fields=externalIds&limit={limit}"
        headers = {"Content-Type": "application/json"}
        if self.s2_key:
            headers["x-api-key"] = self.s2_key
        self.stats.s2_calls += 1
        payload = _http_json(urllib.request.Request(url, headers=headers),
                             timeout=self.timeout_s, stats=self.stats)
        if payload is None:
            return []

        out = []
        for item in payload.get("data", []):
            node = item.get("citedPaper") or item.get("citingPaper") or {}
            ext = (node.get("externalIds") or {})
            if ext.get("ArXiv"):
                out.append(ext["ArXiv"])
        out = list(dict.fromkeys(out))
        self.cache.put("s2", key, out)
        return out

    def citation_counts(self, paper_ids: list[str]) -> dict[str, int]:
        """S2 배치로 피인용수를 한 번에 받습니다 (500개씩).

        응답은 **입력과 같은 순서**로 오고 모르는 논문 자리에는 null 이 옵니다
        (자리가 밀리지 않습니다) — SurveyForge `fetch_citations.py` 가 확인한 동작입니다.
        """
        bases = [strip_version(p) for p in paper_ids]
        out: dict[str, int] = {}
        for i in range(0, len(bases), 500):
            chunk = bases[i : i + 500]
            key = "|".join(chunk[:3]) + f"|n={len(chunk)}|{hashlib.sha1(''.join(chunk).encode()).hexdigest()[:8]}"
            cached = self.cache.get("s2cc", key)
            if cached is not None:
                self.stats.cache_hits += 1
                out.update(cached)
                continue
            if self.offline:
                continue

            headers = {"Content-Type": "application/json"}
            if self.s2_key:
                headers["x-api-key"] = self.s2_key
            req = urllib.request.Request(
                f"{S2_BATCH}?fields=citationCount",
                data=json.dumps({"ids": [f"arXiv:{b}" for b in chunk]}).encode(),
                headers=headers)
            self.stats.s2_calls += 1
            payload = _http_json(req, timeout=self.timeout_s, stats=self.stats)
            if payload is None:
                continue
            got = {b: (p or {}).get("citationCount", 0) or 0 for b, p in zip(chunk, payload)}
            self.cache.put("s2cc", key, got)
            out.update(got)
        return out


# --- 파싱 유틸 ----------------------------------------------------------------

_ARXIV_ABS_URL = re.compile(r"arxiv\.org/abs/(.+)$")


def _parse_arxiv_atom(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        raw_id = (e.findtext(f"{ATOM}id") or "").strip()
        m = _ARXIV_ABS_URL.search(raw_id)
        if not m:
            continue
        pid = m.group(1)
        cats = [c.get("term") for c in e.findall(f"{ATOM}category") if c.get("term")]
        out.append({
            "paper_id": pid,
            "base_id": strip_version(pid),
            "title": " ".join((e.findtext(f"{ATOM}title") or "").split()),
            "abstract": " ".join((e.findtext(f"{ATOM}summary") or "").split()),
            "date": (e.findtext(f"{ATOM}published") or "")[:10],
            "categories": cats,
        })
    return out


def _to_paper(r: dict) -> Paper:
    return Paper(
        paper_id=r["paper_id"],
        base_id=r.get("base_id") or strip_version(r["paper_id"]),
        title=r.get("title", ""),
        abstract=r.get("abstract", ""),
        date=r.get("date", ""),
        submitted_date=r.get("date", ""),   # arXiv API 의 published 는 v1 게시일입니다
        categories=tuple(r.get("categories") or ()),
        citation_count=r.get("citation_count"),
    )
