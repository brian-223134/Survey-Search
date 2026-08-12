"""P1.1 — 자료형. DESIGN §3 의 구현.

원칙 하나만 기억하면 됩니다: **`stats` 에 남지 않는 폐기는 없습니다.**
필터·윈도우·컷오프가 논문을 버렸으면 몇 편을 왜 버렸는지가 반드시 남아야 합니다.
SurveyForge 에서 날짜 윈도우가 예외도 로그도 없이 논문을 버리던 문제가 실제로 있었습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime


@dataclass(frozen=True)
class Paper:
    """검색 결과 1편. `paper_id` 는 **백엔드의 native id** 를 그대로 돌려줍니다 —
    호스트 에이전트가 자기 DB에서 바로 조회할 수 있어야 하기 때문입니다."""

    paper_id: str                       # "2401.12345v2"
    base_id: str                        # "2401.12345" — 교차 코퍼스 정합 키
    title: str
    abstract: str
    date: str                           # ISO 8601 (YYYY-MM-DD)
    categories: tuple[str, ...] = ()
    citation_count: int | None = None   # 백엔드가 모르면 None (AutoSurvey 백엔드)
    score: float = 0.0                  # 최종 랭킹 점수
    facets: tuple[str, ...] = ()        # 이 논문을 끌어올린 facet들
    provenance: tuple[str, ...] = ()    # {"dense", "bm25", "snowball"}

    def months_since(self, ref: _date | None = None) -> float | None:
        """게시 후 경과 개월. `date` 가 없거나 깨졌으면 None."""
        if not self.date:
            return None
        try:
            d = datetime.strptime(self.date[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
        ref = ref or _date.today()
        return (ref - d).days / 30.44


@dataclass(frozen=True)
class Facet:
    name: str                           # 사람이 읽는 하위 주제명
    queries: tuple[str, ...] = ()       # 이 facet으로 실제 실행한 쿼리들
    paper_ids: tuple[str, ...] = ()     # 이 facet이 기여한 논문 (최종 랭킹 순)


@dataclass
class StageStat:
    """파이프라인 한 단계의 입출력. `dropped` 는 `in_n - out_n` 이 아니라
    **단계가 스스로 센 값**입니다 — 둘이 다르면 그 자체가 버그 신호입니다."""

    name: str
    in_n: int = 0
    out_n: int = 0
    dropped: int = 0
    reason: str = ""                    # 왜 버렸는지 (사람이 읽는 한 줄)
    elapsed_s: float = 0.0
    skipped: bool = False               # 단계를 껐거나 백엔드가 지원 안 함
    note: str = ""

    def __str__(self) -> str:
        if self.skipped:
            return f"{self.name:<12} SKIPPED ({self.note or 'disabled'})"
        drop = f" -{self.dropped:,} ({self.reason})" if self.dropped else ""
        return (
            f"{self.name:<12} {self.in_n:>7,} -> {self.out_n:>7,}{drop}"
            f"  {self.elapsed_s:.2f}s"
        )


@dataclass
class SearchStats:
    """단계별 건수·시간과 최신성 지표.

    `recent_12m_ratio` 가 이 프로젝트의 주 관심 지표입니다 — 인용수 정렬이
    최신 논문을 배제한다는 가설을 이 숫자로 확인합니다.
    """

    topic: str = ""
    backend: str = ""
    stages: list[StageStat] = field(default_factory=list)
    total_s: float = 0.0

    n_queries: int = 0
    n_dense_hits: int = 0
    n_lexical_hits: int = 0
    n_final: int = 0

    # 경로별 기여 — "BM25 가 dense 가 못 잡은 걸 데려왔나"에 답하는 숫자.
    # 이 프로젝트의 주장이 성립하려면 n_bm25_only 가 의미 있게 커야 합니다.
    n_dense_only: int = 0
    n_bm25_only: int = 0
    n_both: int = 0

    recent_6m_ratio: float = 0.0
    recent_12m_ratio: float = 0.0
    recent_24m_ratio: float = 0.0
    date_min: str = ""
    date_max: str = ""
    n_missing_date: int = 0
    n_missing_citation: int = 0

    warnings: list[str] = field(default_factory=list)

    def stage(self, name: str) -> StageStat | None:
        return next((s for s in self.stages if s.name == name), None)

    def add(self, stat: StageStat) -> StageStat:
        self.stages.append(stat)
        return stat

    def warn(self, message: str) -> None:
        """무음 스킵 금지 — 건너뛴 단계·비어 있는 필드는 전부 여기 남습니다."""
        self.warnings.append(message)

    def report(self) -> str:
        lines = [
            f"topic   : {self.topic}",
            f"backend : {self.backend}",
            f"queries : {self.n_queries}   dense_hits={self.n_dense_hits:,} "
            f"lexical_hits={self.n_lexical_hits:,}",
            "",
            *(f"  {s}" for s in self.stages),
            "",
            f"final   : {self.n_final:,} papers in {self.total_s:.1f}s",
            f"source  : dense_only={self.n_dense_only:,}  bm25_only={self.n_bm25_only:,}  "
            f"both={self.n_both:,}",
            f"recency : 6m={self.recent_6m_ratio:.1%}  12m={self.recent_12m_ratio:.1%}  "
            f"24m={self.recent_24m_ratio:.1%}",
            f"dates   : {self.date_min} .. {self.date_max}"
            + (f"  (missing {self.n_missing_date:,})" if self.n_missing_date else ""),
        ]
        if self.n_missing_citation:
            lines.append(f"no cite : {self.n_missing_citation:,} papers")
        if self.warnings:
            lines += ["", "warnings:", *(f"  ! {w}" for w in self.warnings)]
        return "\n".join(lines)


@dataclass(frozen=True)
class SearchResult:
    topic: str
    papers: tuple[Paper, ...]
    facets: tuple[Facet, ...]
    stats: SearchStats

    def __len__(self) -> int:
        return len(self.papers)

    def ids(self) -> list[str]:
        """호스트 에이전트가 쓰는 형태 — native id 목록 (랭킹 순)."""
        return [p.paper_id for p in self.papers]


@dataclass(frozen=True)
class SearchConfig:
    """**이 불리언들이 곧 ablation 축입니다.** 전부 끄면 "토픽 1쿼리 → dense top-k" 가 되어
    AutoSurvey 베이스라인과 (임베딩 모델 차이를 빼면) 같아집니다. 그게 비교의 원점입니다."""

    n_papers: int = 1500

    facets: bool = False        # S1 — P2에서 켭니다
    lexical: bool = True        # S3 — BM25
    freshness: bool = False     # S6 — P2
    diversity: bool = False     # S7 — P2

    #: 단계별 후보 폭. 최종 n_papers 보다 넉넉해야 dedup 후에도 남습니다
    dense_top_k: int = 2000
    lexical_top_k: int = 2000

    rrf_k: int = 60             # DESIGN §S4

    #: S6 설정. None 이면 기본값(alpha=beta=0.5, WEIGHT 모드)
    freshness_config: object | None = None

    #: S1 설정 (FacetConfig). None 이면 기본값 + 환경변수
    facet_config: object | None = None

    #: S5 제목 병합 / S6·S7 랭킹이 실제로 보는 후보 수. None 이면 후보 전체.
    #: facet 을 켜면 후보가 수만 편이 되는데, 여기를 작게 잡으면 S6·S7 이 RRF 상위
    #: 일부만 재정렬하게 되어 **단계의 효과가 구조적으로 축소됩니다.**
    #: 값을 줄이면 그만큼 제외된 건수가 stats.warnings 에 남습니다.
    rank_window: int | None = None
    title_window: int | None = None

    #: S7 MMR 이 실제로 훑는 후보 수. **None 이 아니라 기본값이 있는 이유**:
    #: MMR 은 후보 풀이 커질수록 다양성 쪽으로 쏠립니다. 관련성을 풀 안에서 min-max
    #: 정규화하므로, 4만 편을 넣으면 대부분의 relevance 가 0 근처가 되고 λ 가 의도한
    #: 균형이 깨집니다. 실측: 풀 3,000 → 12개월 44.5%, 풀 48,214 → 18.3%.
    #: 랭킹(S6)은 전량 처리해도 싸지만 MMR 은 풀을 묶어야 합니다.
    mmr_pool: int | None = None   # None 이면 max(n_papers * 2, 3000)

    #: S7 — MMR 의 관련성/다양성 균형. 1.0 = 순수 관련성(= 끈 것과 같음), 0.0 = 순수 다양성
    mmr_lambda: float = 0.7
    #: S7 — facet 당 최소 배정. None 이면 n_papers // facet 수
    min_per_facet: int | None = None

    date_min: str | None = None
    date_max: str | None = None
    categories: tuple[str, ...] | None = None

    #: dense 검색에 쓸 인덱스. "title_abs" | "title"
    dense_field: str = "title_abs"
