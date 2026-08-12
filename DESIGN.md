# DESIGN — survey-search

토픽 → 논문 집합 검색 레이어의 설계. 환경 사실은 [`SETTING.md`](SETTING.md),
작업 순서는 [`TASKS.md`](TASKS.md)에 있습니다.

---

## 1. 설계 원칙

**① 에이전트 비종속** — AutoSurvey · SurveyForge · SurveyX · 향후 에이전트가 각자 다른 코퍼스,
다른 임베딩, 다른 id 체계를 씁니다. 따라서 코어는 어느 코퍼스도 모르고, `Backend` 프로토콜만 압니다.
검색 결과는 **백엔드의 native id를 그대로** 돌려줍니다 — 호스트 에이전트가 자기 DB에서 바로
조회할 수 있어야 하기 때문입니다.

**② 드롭인 어댑터** — 각 에이전트의 기존 호출 시그니처를 그대로 흉내내는 얇은 어댑터를 둡니다.
호스트 코드는 한 줄만 바꿔 끼울 수 있어야 하고, 그래야 "검색만 바꿨을 때의 효과"를 통제 측정할 수 있습니다.

**③ 결정적 파이프라인 + 단계별 stats** — LLM은 facet 분해에만 쓰고 결과를 캐시합니다.
나머지 단계는 전부 결정적이며, 각 단계의 입출력 건수를 `stats`에 남깁니다.
**단계를 끄고 켜서 기여도를 분리 측정할 수 있는 것**이 이 패키지의 존재 이유입니다.

**④ 무음 폐기 금지** — 필터·윈도우·컷오프가 버린 논문 수는 반드시 세어서 보고합니다.
SurveyForge에서 날짜 윈도우가 예외도 로그도 없이 논문을 버리던 문제가 실제로 있었습니다.

## 2. 패키지 구조

```
survey-search/
├── README.md · DESIGN.md · SETTING.md · TASKS.md
├── pyproject.toml
├── src/survey_search/
│   ├── types.py              Paper, Facet, SearchResult, SearchStats
│   ├── backends/
│   │   ├── base.py           Backend 프로토콜
│   │   ├── faiss_duckdb.py   FAISS(dense) + DuckDB(meta·BM25) 범용 구현
│   │   └── registry.py       이름 → 백엔드 팩토리 (설정은 .env/yaml)
│   ├── index/
│   │   ├── build_duckdb.py   TinyDB JSON → DuckDB papers 테이블 + FTS 인덱스
│   │   └── inspect_faiss.py  인덱스 지문·id 매핑 실측 도구
│   ├── core/
│   │   ├── facets.py         S1  topic → facet 쿼리 (LLM + 캐시)
│   │   ├── fuse.py           S4  RRF 융합
│   │   ├── dedup.py          S5  arXiv 버전 병합 + 제목 정규화
│   │   ├── rank.py           S6  freshness-aware 랭킹
│   │   ├── diversity.py      S7  MMR / facet 쿼터
│   │   ├── rerank.py         S8a cross-encoder (후속)
│   │   └── expand.py         S8b 인용 스노우볼링 (후속)
│   ├── tools.py              에이전트가 직접 부를 수 있는 검색 프리미티브
│   ├── search.py             오케스트레이터 — search_topic()
│   ├── adapters/
│   │   ├── autosurvey.py     database.get_ids_from_query 호환
│   │   └── surveyforge.py    GeneralRAG_langchain.retrieve_id 호환
│   └── cli.py                python -m survey_search.cli --topic "..."
└── tests/                    mock 백엔드 기반 단위 테스트
```

## 3. 자료형

```python
@dataclass(frozen=True)
class Paper:
    paper_id: str              # 백엔드의 native id (예: "2401.12345v2")
    base_id: str               # 버전 제거 id — 교차 코퍼스 정합 키
    title: str
    abstract: str
    date: str                  # ISO 8601
    categories: tuple[str, ...]
    citation_count: int | None # 백엔드가 모르면 None (AutoSurvey 백엔드)
    score: float               # 최종 랭킹 점수
    facets: tuple[str, ...]    # 이 논문을 끌어올린 facet들
    provenance: tuple[str, ...] # {"dense", "bm25", "snowball"} — 어느 경로로 들어왔나

@dataclass(frozen=True)
class Facet:
    name: str                  # 사람이 읽는 하위 주제명
    queries: tuple[str, ...]   # 이 facet으로 실제 실행한 쿼리들
    paper_ids: tuple[str, ...] # 이 facet이 기여한 논문 (최종 랭킹 순)

@dataclass(frozen=True)
class SearchResult:
    topic: str
    papers: tuple[Paper, ...]  # 최종 랭킹
    facets: tuple[Facet, ...]
    stats: SearchStats
```

`SearchStats`는 단계별 in/out 건수, 각 단계 소요 시간, 버려진 건수와 사유,
그리고 **최근 12/24개월 논문 비율**을 담습니다. 마지막 항목이 이 프로젝트의 주 관심 지표입니다.

### Backend 프로토콜

백엔드가 구현해야 하는 것은 이게 전부입니다:

```python
class Backend(Protocol):
    name: str
    def dense_search(self, queries: list[str], top_k: int,
                     field: str = "title_abs") -> list[list[tuple[str, float]]]: ...
    def lexical_search(self, queries: list[str], top_k: int) -> list[list[tuple[str, float]]]: ...
    def get_papers(self, paper_ids: list[str]) -> list[Paper]: ...
    def filter_ids(self, *, date_min=None, date_max=None,
                   categories=None) -> set[str] | None: ...
    # 선택: 없으면 해당 단계를 건너뜁니다
    def references(self, paper_id: str) -> list[str]: ...
    def cited_by(self, paper_id: str) -> list[str]: ...
```

- `dense_search`/`lexical_search`는 **배치**입니다 — facet fan-out이 기본 사용 패턴이라
  쿼리 하나씩 도는 API는 처음부터 만들지 않습니다.
- `references`/`cited_by`가 없는 백엔드에서는 스노우볼링 단계가 자동으로 no-op이 되고,
  그 사실이 `stats`에 남습니다(무음 스킵 금지).

## 4. 파이프라인

```
topic
  │
  ├─ S1 facet 분해 ────────── LLM 1회 (캐시) → facet 8~16개, 각 1~3 쿼리
  │
  ├─ S2 dense 검색 ────────── facet별 top-K  (title+abs 인덱스, 필요시 title 인덱스도)
  ├─ S3 lexical 검색 ──────── facet별 top-K  (DuckDB BM25)
  │
  ├─ S4 RRF 융합 ─────────── facet 내부 융합 → facet 간 융합
  ├─ S5 중복 제거 ─────────── arXiv 버전 병합, 제목 정규화
  ├─ S6 freshness 랭킹 ────── 연령 정규화 인용률 + recency 가중
  ├─ S7 다양성 ───────────── MMR + facet 쿼터
  │
  ├─ S8 (후속) 스노우볼링 · cross-encoder 재랭킹
  │
  └─ SearchResult (papers + facets + stats)
```

각 단계는 **끌 수 있고**(`SearchConfig`의 불리언), 껐을 때의 기본 동작이 정의돼 있습니다.

### S1 — facet 분해

토픽 문자열 하나로는 dense 이웃 한 덩어리밖에 못 봅니다. LLM에게 토픽을
**하위 주제(facet) 8~16개**로 쪼개게 하고, 각 facet마다 표현이 다른 쿼리 1~3개를 받습니다.
동의어·약어·구식 표기를 섞도록 프롬프트에서 요구합니다(예: "RAG" / "retrieval-augmented generation" /
"retrieval augmented LM").

- 출력은 `{topic_hash}.json`으로 캐시 — 같은 토픽 재실행은 LLM 호출 0회, 완전 결정적
- **끄면**: 토픽 문자열 자체가 유일한 쿼리 (= AutoSurvey 베이스라인과 동등)
- 주의: 이 단계는 LLM의 **사전 지식**에 의존하므로 모델 컷오프 이후의 신조어는 못 냅니다.
  이 한계를 S3(BM25)와 S8(스노우볼링)이 보완하는 구조입니다.

### S2 — dense 검색

기존 FAISS 인덱스를 그대로 씁니다(재임베딩 없음). 쿼리 임베딩만 gte-large-en-v1.5로 계산합니다.

- 인덱스가 `IndexFlatL2` 계열이라 **브루트포스**입니다 — 쿼리당 지연을 P0에서 실측하고,
  느리면 HNSW 변환 또는 GPU FAISS로 전환합니다([SETTING.md](SETTING.md) §6-B)
- 메트릭이 L2인지 IP인지에 따라 점수 방향이 반대이므로, 융합 전에 **순위**로 바꿔 씁니다
  (RRF가 점수 스케일에 무관한 것도 이 때문에 선택했습니다)

### S3 — lexical 검색 (BM25)

DuckDB FTS로 title/abstract에 BM25를 겁니다. **dense가 구조적으로 못 잡는 것**을 담당합니다:
새 방법론 이름, 모델명, 데이터셋명, 약어 — 즉 최신 논문을 식별하는 바로 그 토큰들입니다.

- SurveyForge는 이 자리가 `NotImplementedError`이고, SimScholarSearch는 FTS 인덱스를 만들어 놓고
  어떤 툴에도 연결하지 않았습니다. 양쪽 다 비어 있는 자리입니다.
- **끄면**: dense 결과만 융합 (기여도 측정용 ablation)

### S4 — RRF 융합

`score(d) = Σ_q 1 / (k + rank_q(d))`, k=60.

점수 스케일이 다른 두 검색(코사인/L2 vs BM25)과 여러 facet 쿼리를 **순위만으로** 합칩니다.
2단으로 적용합니다: ① facet 내부에서 dense+BM25 융합 → ② facet 간 융합.
2단으로 나누는 이유는 facet별 쿼터(S7)를 걸려면 facet 소속 정보가 살아 있어야 하기 때문입니다.

### S5 — 중복 제거

1. **arXiv 버전 병합** — `2401.12345v1`과 `v2`는 같은 논문. 최신 버전을 대표로 남기고 점수는 최대값
2. **제목 정규화 일치** — 소문자화·공백/구두점 제거 후 동일하면 병합 (재게시·크로스리스트)

버린 건수를 `stats.dedup_dropped`에 기록합니다.

### S6 — freshness-aware 랭킹

**이 프로젝트의 핵심 가설이 들어가는 자리입니다.**

문제: 인용수 정렬은 최근 논문을 구조적으로 배제합니다. 2026-06 논문은 아무리 중요해도
인용수가 0에 가깝습니다.

제안하는 점수:

```
final = rrf_score × (1 + α · citation_rate_percentile) × (1 + β · recency_weight)
citation_rate = citation_count / max(months_since_pub, 3)      # 연령 정규화
```

- `citation_rate`를 **같은 연령대 논문 안에서의 백분위**로 변환해 씁니다.
  절대 인용수 대신 "또래 대비 얼마나 빨리 인용되는가"를 보는 것입니다.
- `recency_weight`는 최근 N개월에 완만한 가산. 또는 **최근 12개월 쿼터**(최종 목록의 x%를
  최근 논문에 강제 배정)로 대체 가능 — 둘 다 구현하고 비교합니다.
- `citation_count`가 없는 백엔드(AutoSurvey)에서는 인용 항이 자동으로 1이 되고 `stats`에 남습니다.
- **끄면**: RRF 점수 그대로 (SurveyForge의 `citation` 재랭킹과 A/B 비교 대상)

### S7 — 다양성

서베이의 목적 함수는 정확도가 아니라 **커버리지**입니다. 상위권이 한 연구 그룹·한 계열로
쏠리면 서베이의 섹션 하나가 통째로 비게 됩니다.

- **MMR**: `λ · relevance − (1−λ) · max_sim(선택된 것들)`, 유사도는 초록 임베딩 기준
- **facet 쿼터**: 최종 N편을 facet 수로 나눠 최소 배정 보장. facet이 균등하지 않을 수 있어
  최소 보장만 하고 나머지는 점수순
- **끄면**: 점수순 상위 N

### S8 — 후속 단계 (P4 이후)

- **인용 스노우볼링** — 상위 시드 20~50편의 전방/후방 인용 2-hop. arXiv 메타데이터에는
  인용 엣지가 없으므로 Semantic Scholar / OpenAlex API 보강이 선행돼야 합니다
  (SurveyForge가 `citation_count`를 S2에서 채운 전례가 있으니 경로는 뚫려 있습니다)
- **cross-encoder 재랭킹** — top-500 → top-100. `BAAI/bge-reranker-v2-m3` 등. 별도 다운로드 필요

## 5. 오케스트레이터 API

```python
from survey_search import search_topic, SearchConfig

result = search_topic(
    "Retrieval-Augmented Generation for Large Language Models",
    backend="surveyforge-2026-08",
    config=SearchConfig(
        n_papers=1500,
        facets=True, lexical=True, freshness=True, diversity=True,
        date_min=None, date_max=None,
    ),
)
result.papers[0].title
result.stats.recent_12m_ratio
```

`SearchConfig`의 불리언들이 곧 ablation 축입니다. 전부 끄면 "토픽 1쿼리 → dense top-k"가 되어
AutoSurvey 베이스라인과 (임베딩 모델 차이를 빼면) 같아집니다. 이게 비교의 원점입니다.

## 6. 어댑터 규약

호스트 에이전트는 **한 줄만 바꿔 끼울 수 있어야** 합니다.

**AutoSurvey** — [`src/database.py:86`](../AutoSurvey/src/database.py#L86)의
`get_ids_from_query(query, num, shuffle=False) -> list[str]`

```python
from survey_search.adapters.autosurvey import SurveySearchDatabase
db = SurveySearchDatabase(db_path, embedding_model)   # 기존 database()와 동일 시그니처
```

**SurveyForge** — [`code/src/rag.py:227`](../SurveyForge/code/src/rag.py#L227)의
`retrieve_id(query, search_type, rerank, top_k, max_out, filter, fetch_k) -> list[str]`

`filter`(인덱스 서브셋 제한)와 `rerank='citation'`을 그대로 받아야 기존 2단계 구조가 유지됩니다.
어댑터는 이를 `SearchConfig`로 번역합니다.

**신규 에이전트** — 어댑터 없이 `search_topic()`을 직접 부르고 `SearchResult`를 씁니다.
어댑터는 어디까지나 **기존 코드를 안 고치기 위한 호환 계층**이며, 코어 API가 정본입니다.

## 7. 열려 있는 결정

| # | 결정 | 지금 시점의 방향 |
|---|---|---|
| 1 | 백엔드 간 코퍼스 통합 여부 | **하지 않음.** 백엔드별로 독립 유지하고 `base_id`로만 교차 참조. 통합은 재임베딩을 부르고, 그러면 베이스라인과의 통제 비교가 깨짐 |
| 2 | 임베딩 모델 통일 | 1차는 **통일하지 않음** — 기존 인덱스 재사용이 목적. BGE-M3 재구축은 P5에서 별도 축으로 |
| 3 | facet 분해에 쓸 LLM | 기존 레포가 쓰는 OpenRouter 키 재사용. 캐시가 있어 비용은 토픽당 1회 |
| 4 | 평가 | SurGE 코드 미공개 → 우선 **내부 진단 지표**(stats, 최신 논문 비율, facet 커버리지)로 개발. 정량 벤치마크는 후속 |
