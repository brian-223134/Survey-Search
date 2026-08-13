# survey-search

토픽 하나를 받아 관련 논문 집합을 찾아 주는 **검색 레이어**.
AutoSurvey · SurveyForge · SurveyX 등 여러 서베이 에이전트가 공유해서 쓰는 독립 패키지입니다.

```
topic: str  →  ranked · deduped · facet-grouped papers
```

**최종 갱신**: 2026-08-13 · 테스트 154개 통과

---

## 1. 무엇을 풀려고 하는가

서베이 생성 에이전트는 전부 "논문을 찾아서 → 개요를 짜고 → 쓴다"는 구조입니다. 그런데
**첫 단계인 검색은 거의 손대지 않은 채로 남아 있습니다.**

기존 구현을 열어 보면 이렇습니다.

| | AutoSurvey | SurveyForge |
|---|---|---|
| 쿼리 | 토픽 문자열 1개 → top-1200 | 멀티쿼리 union + 2단계(1500편 서브셋) |
| 융합 | 없음 | union — **순위 정보를 버림** |
| 어휘 검색 | 없음 | `NotImplementedError` 스텁 |
| 재랭킹 | 없음 | 2년 시간창별 **인용수 정렬** |
| 인용 그래프 | 없음 | 없음 |

여기서 두 가지 문제가 생깁니다.

**① 데이터 신선도** — 배포본 DB의 컷오프가 2024-04(AutoSurvey) / 2024-09(SurveyForge)였습니다.
2026-08 스냅샷으로 최신화하는 작업은 이미 끝났습니다.

**② 랭킹이 최신 논문에 구조적으로 불리함** — DB만 최신화해서는 해결되지 않습니다.
SurveyForge의 `sort_by_citation_period`는 시간창 안에서 **인용수로 정렬**합니다. 최근
6~12개월 논문은 인용수가 구조적으로 0에 가까워, 검색에 걸려도 랭킹에서 다시 탈락합니다.
AutoSurvey는 재랭킹이 없는 대신 토픽 임베딩의 최근접 이웃이 전부라, 새로 등장한 하위
분야는 **용어 자체가 새로워서** dense 이웃에 잡히지 않습니다.

**전제는 데이터로 확인됐습니다.** 코퍼스 908,819편 중 2025~2026년 논문이 277,804편
(**30.6%**)인데 인용수 중앙값은 6입니다. 인용수 정렬은 코퍼스의 3할을 구조적으로
뒤로 보냅니다.

## 2. 어떻게 접근했는가

검색을 **끄고 켤 수 있는 단계로** 쌓았습니다. 각 단계의 기여를 분리 측정할 수 있어야
"무엇이 실제로 효과가 있는가"에 답할 수 있기 때문입니다.

```
topic
  ├─ S1  facet 분해 ──── LLM 1회(캐시) → 하위 주제 8~16개, 각 1~3 쿼리
  ├─ S2  dense 검색 ──── FAISS (gte-large-en-v1.5, 1024d, 908,819편)
  ├─ S3  어휘 검색 ────── DuckDB BM25
  ├─ S4  RRF 융합 ────── facet 내부 → facet 간, 2단
  ├─ S5  중복 제거 ────── arXiv 버전 병합 + 제목 정규화
  ├─ S6  freshness ───── 연령 정규화 인용률 백분위 + recency
  ├─ S8  스노우볼링 ───── S2 API 인용 그래프 (전방·후방)
  ├─ S8b cross-encoder ─ 재랭킹 (측정 결과 권장하지 않음)
  ├─ S7  다양성 ──────── MMR + facet 쿼터 (측정 결과 권장하지 않음)
  └─ SearchResult (papers + facets + stats)
```

설계 원칙 네 가지:

- **에이전트 비종속** — 코어는 어느 코퍼스도 모르고 `Backend` 프로토콜만 압니다.
  결과는 백엔드의 native id 를 그대로 돌려줍니다
- **드롭인 어댑터** — 호스트 코드를 한 줄만 바꿔 끼울 수 있어야 "검색만 바꿨을 때의
  효과"를 통제 측정할 수 있습니다
- **결정적 파이프라인** — LLM은 facet 분해에만 쓰고 결과를 캐시합니다. 온라인 호출도
  전부 캐시합니다. 같은 입력이면 같은 결과입니다
- **무음 폐기 금지** — 필터·윈도우·컷오프가 버린 논문 수는 반드시 세어서 `stats` 에
  남깁니다. 이 원칙이 실제로 버그를 다섯 번 잡았습니다(§8)

## 3. 설치와 사용

### 요구사항

| | |
|---|---|
| Python | 3.10+ |
| 인덱스 | SurveyForge 2026-08 스냅샷 (FAISS 3.7GB × 2 + TinyDB 1.4GB) |
| GPU | 쿼리 임베딩용. 없으면 CPU 로 자동 폴백(느리지만 동작) |
| 디스크 | DuckDB 산출물 1.1GB |

### 설치

```bash
git clone https://github.com/brian-223134/Survey-Search.git survey-search
cd survey-search

virtualenv -p python3.10 .venv          # python3 -m venv 가 막힌 환경이면 이걸로
.venv/bin/pip install -e .

# GPU 를 쓸 경우 드라이버에 맞는 torch 를 고정하세요.
# CUDA 12.4 (드라이버 550.x) 예시:
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

> torch 기본 휠은 최신 드라이버를 요구합니다. 맞지 않으면 GPU 를 못 잡고 CPU 로
> 폴백하는데, 그 사실이 로그에 남으니 확인하세요.

### 인덱스 준비

`SURVEY_SEARCH_SF_DB` 로 FAISS/TinyDB 위치를 알려 준 뒤 DuckDB 를 만듭니다 (약 90초).

```bash
export SURVEY_SEARCH_SF_DB=/path/to/database_2026-08
.venv/bin/python -m survey_search.index.build_duckdb      # -> data/papers.duckdb

# 인덱스가 멀쩡한지 확인 (id 매핑 왕복 + 지연 측정)
.venv/bin/python -m survey_search.index.inspect_faiss --probe 10
.venv/bin/python -m pytest tests/ -q
```

### 환경변수

`.env.example` 을 복사해 채웁니다. **키가 없어도 검색은 돌아갑니다** — facet 분해만
규칙 기반으로 내려가고 그 사실이 `stats` 에 남습니다.

```bash
cp .env.example .env && chmod 600 .env
```

| 변수 | 필요한 때 |
|---|---|
| `OPENROUTER_API_KEY` | S1 facet 분해 (토픽당 1회, 캐시됨) |
| `SEMANTIC_SCHOLAR_API_KEY` | S8 스노우볼링 / 온라인 백엔드 |
| `SURVEY_SEARCH_SF_DB` | 인덱스 경로 (기본값이 맞으면 불필요) |
| `SURVEY_SEARCH_DATA_DIR` | 산출물 경로 (기본 `./data`) |

### 쓰는 법 — 새 프로젝트

```python
from survey_search.core.facets import load_dotenv; load_dotenv(".env")
from survey_search.backends.faiss_duckdb import FaissDuckDBBackend
from survey_search.search import search_topic
from survey_search.types import SearchConfig

backend = FaissDuckDBBackend()      # 재사용하세요. 첫 검색만 12초(인덱스 로드), 이후 0.1초

result = search_topic(
    "Retrieval-Augmented Generation for Large Language Models",
    backend=backend,
    config=SearchConfig(n_papers=1500, facets=True, freshness=True),   # 권장 기본값
)

result.ids()                  # ['2401.12345v2', ...] 랭킹 순
result.papers[0].title        # Paper: paper_id/base_id/title/abstract/date/
                              #        submitted_date/categories/citation_count/
                              #        score/facets/provenance
result.facets                 # facet별 이름·쿼리·기여 논문
print(result.stats.report())  # 단계별 in/out, 폐기 건수와 사유, 최신성 비율
```

### 쓰는 법 — CLI

```bash
.venv/bin/python -m survey_search.cli --topic "..." --facets --freshness --out results/
.venv/bin/python -m survey_search.cli --topics-file topics.txt --all --out results/
.venv/bin/python -m survey_search.cli --topic "..." --ablate --out results/   # 설정별 비교표
```

### 쓰는 법 — 기존 에이전트에 끼우기

호스트 코드는 그대로 두고 import 한 줄만 바꿉니다.

```python
# AutoSurvey — src/database.py 의 database 대체
from survey_search.adapters.autosurvey import SurveySearchDatabase as database

# SurveyForge — code/src/rag.py 의 GeneralRAG_langchain 대체
from survey_search.adapters.surveyforge import SurveySearchRAG
rag = SurveySearchRAG(backend=FaissDuckDBBackend())
```

- `retrieve_id(query, rerank, filter, top_k, max_out)` 시그니처를 그대로 받습니다
- `rerank='citation'` 은 **freshness 랭킹으로 대체**되고, 그 사실이
  `rag.last_stats.warnings` 에 남습니다. A/B 비교 보고에 반드시 명시하세요
- AutoSurvey 어댑터는 **SurveyForge 스냅샷의 id** 를 돌려줍니다. 모든 논문 조회가
  어댑터를 거쳐야 하고, 본문(`paper_content.h5`)을 읽는 경로는 id 가 안 맞아
  `NotImplementedError` 를 냅니다

### 설정 스위치 = 실험 축

| 옵션 | 기본 | 권장 | 하는 일 |
|---|---|---|---|
| `facets` | `False` | **켜기** | S1. LLM 이 토픽을 하위 주제로 분해. **기여가 가장 큽니다** |
| `lexical` | `True` | 켜기 | S3. BM25. 꼬리 recall 을 늘립니다 |
| `freshness` | `False` | **켜기** | S6. 연령 정규화 인용률 + recency |
| `diversity` | `False` | **끄기** | S7. MMR. **정답 기준으로 항상 손해였습니다**(§7) |
| `rerank` | `False` | **끄기** | S8b. cross-encoder. **세 가지 쿼리 방식 모두 기준선 미달**(§7) |
| `snowball` | `False` | 선택 | S8. 인용 그래프 확장. 온라인/하이브리드 백엔드 필요 |
| `n_papers` | `1500` | — | 최종 반환 편수 |
| `date_min` / `date_max` | `None` | — | 날짜 컷오프 (**제출일 기준**) |

### 온라인 확장 (선택)

로컬 스냅샷이 못 하는 두 가지 — 컷오프 이후 논문과 인용 엣지 — 를 메웁니다.

```python
from survey_search.backends.online import OnlineBackend
from survey_search.backends.hybrid import HybridBackend
from survey_search.core.expand import SnowballConfig

backend = HybridBackend(FaissDuckDBBackend(), OnlineBackend())
result = search_topic(topic, backend=backend, config=SearchConfig(
    n_papers=1500, facets=True, freshness=True, snowball=True,
    snowball_config=SnowballConfig(n_seeds=15, max_new=800)))
```

응답은 전부 디스크에 캐시되므로 **두 번째 실행부터 네트워크 0회**이고 결과가 재현됩니다.
arXiv 는 초당 1회 제한을 지키느라 쿼리당 3초가 듭니다(facet 36쿼리면 2분).

## 4. 구조

```
src/survey_search/
├── types.py            Paper · Facet · SearchConfig · SearchStats
├── assets.py           자산 경로·상수 한 곳 모음 (경로 하드코딩 방지)
├── search.py           오케스트레이터 — search_topic()
│
├── backends/           코퍼스를 아는 유일한 계층
│   ├── base.py         Backend 프로토콜 (필수 4 + 선택 2)
│   ├── faiss_duckdb.py FAISS(dense) + DuckDB(BM25·메타)  ← 1차 백엔드
│   ├── online.py       arXiv API + Semantic Scholar (인용 엣지)
│   └── hybrid.py       로컬 + 온라인 합성
│
├── core/               파이프라인 단계. 각각 독립적으로 끌 수 있음
│   ├── facets.py       S1  토픽 → facet (LLM + 디스크 캐시)
│   ├── fuse.py         S4  RRF
│   ├── dedup.py        S5  버전 병합 + 제목 정규화
│   ├── rank.py         S6  freshness (연령 정규화 인용률 + recency)
│   ├── expand.py       S8  인용 스노우볼링
│   ├── rerank.py       S8b cross-encoder 재랭킹
│   └── diversity.py    S7  MMR + facet 쿼터
│
├── adapters/           호스트 에이전트 호환 계층 (정본 아님)
│   ├── autosurvey.py   database 드롭인
│   └── surveyforge.py  GeneralRAG_langchain 드롭인
│
├── index/              자산 준비·검증
│   ├── inspect_faiss.py  id 매핑 실측, 지연 측정
│   └── build_duckdb.py   TinyDB JSON → DuckDB + FTS
│
├── metrics/            지표
│   ├── diagnostics.py  정답 없이 재는 진단 (최신성·커버리지·대조)
│   └── paper_set.py    정답 있을 때의 채점 (recall·nDCG)
│
├── eval/               정량 평가
│   ├── surge.py        SurGE 정답 집합 구축 + ablation
│   └── ceiling.py      검색 병목인가 랭킹 병목인가
│
scripts/                측정 실행 스크립트 (재현용)
├── run_surge_eval.py       설정별 recall/nDCG
├── probe_retrieval_depth.py  top_k 별 풀 recall
├── eval_depth_final.py     top_k 2000 vs 8000 최종
├── eval_rerank.py          재랭킹 TOPIC 모드
└── eval_rerank_facet.py    재랭킹 FACET / FACET_MAX
│
└── cli.py              토픽 → JSON, --ablate 비교표
```

의존 방향은 한쪽입니다. **`core/` 는 백엔드를 모르고, 백엔드는 `core/` 를 모릅니다.**
둘을 아는 것은 `search.py` 하나뿐입니다. 그래서 백엔드를 갈아 끼워도 파이프라인이
그대로이고, 단계를 추가해도 백엔드를 안 건드립니다.

### Backend 프로토콜

백엔드가 구현할 것은 이게 전부입니다.

```python
def dense_search(queries: list[str], top_k: int, field: str) -> list[list[Hit]]
def lexical_search(queries: list[str], top_k: int)             -> list[list[Hit]]
def get_papers(paper_ids: list[str])                           -> list[Paper]
def filter_ids(*, date_min, date_max, categories)              -> set[str] | None

# 선택 — 없으면 해당 단계가 no-op 이 되고 그 사실이 stats 에 남습니다
def references(paper_id: str) -> list[str]
def cited_by(paper_id: str)   -> list[str]
```

두 가지가 의도적입니다.

**① 검색이 배치입니다.** facet fan-out 이 기본 사용 패턴이고, 실측상 배치가 쿼리당
9배 빠릅니다(1쿼리 790ms vs 32쿼리 배치 85ms/쿼리). 쿼리 하나씩 도는 API 는 아예
만들지 않았습니다.

**② `filter_ids` 의 `None` 과 빈 집합은 뜻이 다릅니다.** `None` = 제한 없음,
`set()` = 조건에 맞는 논문이 하나도 없음. 이걸 섞으면 필터가 조용히 무력화됩니다.

## 5. 각 단계가 하는 일

### S1 facet 분해 — 쿼리 하나로는 이웃 한 덩어리밖에 못 본다

토픽 문자열 하나는 임베딩 공간의 점 하나입니다. 그 근처만 보게 되고, 새로 생긴 하위
분야는 **용어 자체가 새로워서** 그 근처에 없습니다.

LLM 에게 토픽을 하위 주제 8~16개로 쪼개게 하고, facet 마다 표면형이 다른 쿼리를
1~3개 받습니다 — `"RAG"` / `"retrieval-augmented generation"` / `"retrieval augmented LM"`.
방법론·모델·데이터셋 이름을 넣도록 프롬프트에서 요구합니다. 실제로 Self-RAG · IRCoT ·
DPR · FEVER · CodeT5 같은 이름이 나오고, **이게 dense 임베딩이 구조적으로 못 잡는
바로 그 토큰들입니다.**

- 결과는 `{topic_hash}.json` 으로 캐시 → 같은 토픽 재실행은 LLM 호출 0회
- 한계: LLM 의 **사전 지식**에 의존하므로 모델 컷오프 이후 신조어는 못 냅니다.
  그 구멍을 S3(BM25)와 S8(스노우볼링)이 메우는 구조입니다
- 실패하면 규칙 기반으로 내려가되 **그 사실을 stats 에 남깁니다.** 조용히 "토픽 1쿼리"로
  되돌아가면 facet 을 켠 실험과 끈 실험이 구분되지 않습니다

### S2·S3 dense + 어휘 — 서로 다른 것을 놓친다

dense 는 "말이 비슷한 논문", BM25 는 "단어가 겹치는 논문"을 찾습니다. 새 방법론 이름·
모델명·데이터셋명·약어는 후자의 영역이고, **최신 논문을 식별하는 토큰이 정확히 그것들**입니다.
실측: BM25 는 dense 가 전혀 못 찾은 논문 234편을 데려왔습니다.

### S4 RRF — 점수가 아니라 순위로 합친다

```
score(d) = Σ_q  1 / (k + rank_q(d)),   k = 60
```

**이 인덱스에서는 선택이 아니라 필수 조건입니다.** 저장된 문서 벡터는 단위 norm 인데
gte 가 내놓는 쿼리 벡터는 정규화돼 있지 않습니다(norm ≈ 24). 한 쿼리 안의 순위는
멀쩡하지만 **쿼리끼리 점수를 비교할 수 없습니다.** BM25 점수는 dense 와 단위가 아예
다릅니다. 순위만 쓰면 이 문제가 사라집니다.

2단으로 적용합니다: ① facet 내부에서 dense+BM25 융합 → ② facet 간 융합. 나누는 이유는
facet 소속 정보를 살려 S7 의 쿼터를 걸기 위해서이고, 한 번에 다 섞으면 쿼리를 많이 가진
facet 이 결과를 지배하기 때문입니다.

### S5 중복 제거

`2401.12345v1` 과 `v2` 를 병합합니다 — 대표는 최신 버전, 점수는 최대값. 대표와 점수를
따로 정하는 이유는, 버전이 갈리면 같은 논문의 순위가 반토막 나기 때문입니다.
제목 정규화(소문자·영숫자만) 일치도 병합합니다. 실측상 이 코퍼스에 정규화 제목이
겹치는 그룹이 806개 있어 죽은 코드가 아닙니다.

### S6 freshness — 절대 인용수가 아니라 또래 대비 속도

```
final = rrf × (1 + α·citation_rate_percentile) × (1 + β·recency_weight)
citation_rate = citation_count / max(months_since_pub, 3)
```

2026년 논문의 인용수 3 과 2019년 논문의 인용수 3 은 전혀 다른 의미입니다. 그래서
인용률을 **6개월 코호트 안에서의 백분위**로 바꿔 씁니다. 전체를 한 줄로 세우면 오래된
논문이 상위 백분위를 독식합니다.

곱셈인 이유: RRF 점수가 이미 "여러 검색이 얼마나 동의하는가"를 담고 있으므로 freshness 는
그걸 **대체**하는 게 아니라 **조정**해야 합니다. 덧셈이면 관련 없는 최신 논문이 올라옵니다.

recency 는 두 방식을 다 구현해 두고 고르게 했습니다:
**weight**(연속 감쇠, 밀어주되 보장 없음) / **quota**(모든 접두구간에서 비율 보장, 대신 순위 왜곡).
쿼터는 하한이지 상한이 아니므로, 이미 충족되면 아무것도 하지 않습니다(`promoted=0` 은
고장이 아니라 "필요 없었다"는 뜻).

### S8 스노우볼링 — 저자가 직접 선언한 관계

인용은 임베딩 유사도도 어휘 겹침도 아닌 **제3의 신호**입니다. 저자가 "이 논문이 관련
있다"고 명시한 것이고, 사람이 문헌조사하는 방식이기도 합니다.

- **후방(references)** — 시드가 딛고 선 토대. 오래됐지만 확실히 관련 있음
- **전방(cited_by)** — 시드 이후의 후속 연구. **컷오프 이후 논문이 여기서 나옵니다**

여러 시드가 공통으로 가리킨 논문을 우선합니다(`min_seed_support`). 유입 논문의 점수
(시드 지지도)는 RRF 점수와 스케일이 다르므로 **다시 RRF 로 융합**합니다 — 임의로 점수를
깎아 붙이면 유입 논문이 최종 컷에 영원히 못 들어옵니다(§8).

### S8b cross-encoder 재랭킹 — 구현했지만 권장하지 않습니다

bi-encoder(gte)는 쿼리와 문서를 **따로** 인코딩합니다 — 그래야 90만 편을 미리 색인할 수
있지만 둘이 서로를 보지 못합니다. cross-encoder 는 함께 넣어 한 번에 점수를 냅니다.
정확하지만 후보 수만큼 forward 를 돌려야 해서 상위 수천 편에만 씁니다.

천장 측정에서 랭킹 손실이 18.6%p 로 나왔으니 이 단계가 그걸 되찾을 것으로 봤는데,
**세 가지 쿼리 방식이 모두 기준선에 미달했습니다**(§7). 원인은 쿼리 설계와 점수 비교
방식이었지 재랭커 자체가 아닙니다. 코드는 남겨 뒀고 `rerank=False` 가 기본입니다.

`RerankConfig.top_n` 은 **`n_papers` 보다 커야 합니다.** 작으면 최종 *집합* 이 안 바뀌고
순서만 바뀌어서, 켠 실험과 끈 실험이 recall 로 구분되지 않습니다 — 그런 조합이면
`stats` 에 경고가 뜹니다.

### S7 다양성 — 목적 함수가 정확도가 아니라 커버리지

가장 관련 있는 1500편은 서로 매우 비슷합니다. 관련성만 최적화하면 한 연구 그룹·한
계열로 쏠리고, 서베이의 섹션 하나가 통째로 빕니다.

MMR: `λ·relevance − (1−λ)·max_sim(이미 고른 것)`. 유사도는 **인덱스에 저장된 초록
임베딩**을 그대로 씁니다(재임베딩 0). 이미 고른 것과의 최대 유사도를 매번 전부 다시
재면 O(k²n) 인데, 새로 고른 것 하나만 반영하면 O(kn) 입니다.

> ⚠ **측정 결과 이 단계는 켜지 않는 것이 낫습니다.** 방향 지표(유사도·카테고리 수)는
> 낮은 λ 를 선호하지만, 정답 기준으로는 λ=0.9 조차 안 켜는 쪽보다 나쁩니다. λ=0.3 은
> R@50 을 19.4% → 9.4% 로 반토막 냅니다(§7). 기본값 `diversity=False` 를 유지하세요.
> 코드는 남겨 둡니다 — 커버리지가 목적 함수인 다른 설정에서는 다를 수 있습니다.

MMR 풀은 일부러 묶습니다(`mmr_pool`, 기본 `max(n×2, 3000)`). 관련성을 풀 안에서
min-max 정규화하므로 풀이 커지면 대부분의 relevance 가 0 근처가 되고 다양성이
관련성을 압도합니다. 실측(갱신일 기준으로 잰 값이라 §5 표와 절대값은 다릅니다):
풀을 3,000 에서 48,214 로 키우자 최근 12개월 비율이 45.8% → 18.3% 로 무너졌습니다.

### stats — 모든 단계가 자기가 버린 것을 신고한다

```
S2 dense          36 ->  72,000  17.07s
S4 rrf       144,000 ->  48,257  0.21s
S5 dedup      48,257 ->  48,214 -43 (version=28, title=15)  1.06s
S7 diversity   3,000 ->   1,500 -1,500 (n_papers=1500)  0.14s
warnings:
  ! MMR 풀을 3,000편으로 제한 — 45,214편은 S7 대상에서 제외
```

`dropped` 는 `in - out` 을 계산한 값이 **아니라 단계가 스스로 센 값**입니다. 둘이
다르면 그 자체가 버그 신호입니다. 껐거나 백엔드가 지원 안 해서 건너뛴 단계도
`skipped=True` 로 남습니다 — 껐을 때와 고장 났을 때가 구분돼야 하기 때문입니다.

## 6. 결과 — 방향 지표

토픽 "RAG for LLMs", 1500편 기준. 논문 나이는 arXiv id 에서 유도한 **제출일 기준**입니다.

| 설정 | 쿼리 | 최근 6m | 최근 12m | 상위200 유사도 | 카테고리 | 인용 중앙값 | 신규 |
|---|---|---|---|---|---|---|---|
| dense only (베이스라인) | 1 | 10.8% | 25.1% | 0.7969 | 23 | 4 | — |
| + BM25 | 1 | 10.9% | 25.6% | 0.7545 | 27 | 3 | 351 |
| + freshness | 1 | 12.7% | 28.7% | 0.7574 | 29 | 4 | 361 |
| + 다양성 (MMR λ=0.3) | 1 | 14.7% | 30.8% | 0.6310 | 32 | 2 | 862 |
| + facets | 36 | 14.7% | 33.7% | 0.7653 | 25 | 4 | 744 |
| **전부 켜기** | 36 | **15.5%** | **34.9%** | **0.6302** | 31 | 3 | **1,022** |

최근 12개월 비율 25.1% → 34.9%(+9.8%p), 상위 200편의 평균 쌍별 유사도 0.797 → 0.630,
최종 1,500편 중 1,022편(68%)이 교체됐습니다.

> ⚠ 이 표는 **방향 지표**입니다. "얼마나 최신인가"·"얼마나 다양한가"만 재고 **"얼마나
> 맞는가"는 재지 않습니다.** 실제로 §7 의 정답 기준 평가는 **다양성(MMR)에 대해
> 정반대 결론**을 냈습니다 — 이 표에서 가장 좋아 보이는 '전부 켜기' 가 정답 기준으로는
> facet 만 켠 것보다 나쁩니다. 두 표를 반드시 같이 읽으세요.

## 7. 정량 평가 — 정답 집합으로 재기

문서 초안 단계에서는 "SurGE 벤치마크 구현이 미공개라 정량 평가 보류"로 적어 두었는데,
**직접 확인하니 그 전제가 틀렸습니다.**

[SurGE](https://github.com/oneal2000/SurGE) 의 `data/surveys.json` 에 GT 서베이 205편이 있고, 각각 `survey_title`(토픽)과
`all_cites`(그 서베이가 실제로 인용한 논문 목록)를 가집니다. **서베이가 인용한 논문
집합이 곧 그 토픽의 정답입니다.** README 가 언급하는 `queries.json` 은 배포본에 없지만
없어도 됩니다.

우리 코퍼스와의 연결은 정규화 제목으로 합니다(SurGE 코퍼스에는 arXiv id 가 없습니다):

```
인용 13,485편 중 12,324편(91.4%) 연결  |  서베이별 매칭률 중앙값 92.1%
205편 중 170편 사용 (all_cites 없음 9편, 정답 10편 미만 26편 제외)
```

**날짜 컷오프가 필수입니다.** GT 서베이는 2019~2023년입니다. 2020년 서베이는 2021년
논문을 인용할 수 없으므로, 컷오프 없이 평가하면 우리 파이프라인이 최신 논문을 밀어주는
만큼 자동으로 깎입니다. 서베이마다 `date_max=<게시일>` 을 겁니다.

```bash
python -m survey_search.eval.surge --build-gold    # 정답 캐시 (1회, 약 3분)
python scripts/run_surge_eval.py --limit 16        # 설정별 recall/nDCG
python -m survey_search.eval.ceiling --limit 25    # 검색 병목인가 랭킹 병목인가
```

### 결과 (토픽 16개, 평균 정답 87편, n_papers=1500)

| 설정 | R@50 | R@100 | R@500 | R@1500 | nDCG |
|---|---|---|---|---|---|
| dense only | 16.4% | 21.2% | 37.0% | 39.0% | 0.189 |
| + BM25 | 13.4% | 19.9% | 34.7% | **42.4%** | 0.189 |
| + freshness | 13.9% | 20.6% | 35.8% | **42.4%** | 0.218 |
| **+ facets** | **19.4%** | **24.6%** | **45.1%** | **59.1%** | **0.288** |
| + MMR λ=0.9 | 19.2% | 24.1% | 44.6% | 56.5% | 0.285 |
| + MMR λ=0.7 | 18.4% | 23.7% | 40.1% | 52.1% | 0.279 |
| + MMR λ=0.3 | 9.4% | 14.0% | 20.1% | 38.3% | 0.182 |

**① facet 분해가 압도적입니다.** R@1500 39.0% → 59.1%, nDCG 0.189 → 0.288(+52%).
방향 지표에서도 기여가 가장 컸는데 정답 기준으로도 그렇습니다. LLM 이 내놓는
방법론·데이터셋 이름이 dense 임베딩의 사각지대를 정확히 메운다는 가설이 확인됩니다.

**② 다양성(MMR)은 정답 기준으로 항상 손해였습니다.** λ 를 0.9 까지 올려도 안 켜는
쪽이 낫고, 0.3 은 R@50 을 19.4% → 9.4% 로 반토막 냅니다. **방향 지표(유사도 0.63,
카테고리 32개)는 정반대를 가리켰습니다** — 이 프로젝트에서 방향 지표만 믿으면 안
된다는 것을 보여 주는 가장 분명한 사례입니다. 기본값을 `diversity=False` 로 둡니다.

**③ BM25 는 머리를 깎고 꼬리를 늘립니다.** R@50 은 16.4% → 13.4% 로 내려가는데
R@1500 은 39.0% → 42.4% 로 올라갑니다. 서베이는 수백~수천 편을 모으는 작업이므로
꼬리 recall 이 더 중요하다고 보고 켜 둡니다. 다만 상위권만 쓰는 용도라면 끄는 게 낫습니다.

**④ freshness 는 recall 을 안 건드리고 순서를 개선합니다.** R@1500 은 42.4% 로 같은데
nDCG 가 0.189 → 0.218 로 오릅니다. 같은 논문을 더 앞에 놓는다는 뜻입니다.

### top_k 인상과 재랭킹 — 둘 다 기대와 달랐습니다

**① `top_k` 를 4배 키워도 최종 recall 은 거의 안 오릅니다** (토픽 16개)

| 설정 | R@500 | R@1500 | nDCG |
|---|---|---|---|
| facets `top_k=2000` | 45.1% | 59.1% | 0.288 |
| facets `top_k=8000` | 45.5% | **60.2%** | 0.298 |

풀 recall 은 84.6% → 91.1% (+6.5%p) 올랐는데 최종은 **+1.1%p** 입니다. 새로 데려온
정답이 RRF 순위에서 1,500위 아래에 깔립니다. **랭킹이 병목이라는 것이 두 번째 방향에서
확인됐습니다** (첫 번째는 천장 측정의 랭킹 손실 18.6%p).

**② cross-encoder 재랭킹은 오히려 해로웠습니다**

| 설정 | R@50 | R@500 | R@1500 | nDCG |
|---|---|---|---|---|
| facets (기준) | **19.4%** | **45.1%** | **59.1%** | **0.288** |
| + rerank (`top_n=3000`) | 14.0% | 36.6% | 56.3% | 0.190 |
| + rerank + `top_k=8000` | 14.1% | 36.6% | 55.4% | 0.192 |

nDCG 가 0.288 → 0.190 으로 **34% 떨어집니다.** 한 토픽을 열어 보면 이유가 분명합니다
(「A Survey on Explainability in Machine Reading Comprehension」, R@100 28.9% → 8.4%):

```
재랭킹이 끌어올린 것            재랭킹이 밀어낸 정답
  A Survey on Machine Reading…    HotpotQA (7위 → 940위)
  Neural MRC: Methods and Trends  R3 Benchmark (10위 → 36위)
  ML Interpretability: A Science… Select, Answer and Explain (1위 → 135위)
  Causal Interpretability for ML   Multi-hop RC via Question Decomp. (5위 → 271위)
```

**쿼리가 틀렸지 재랭커가 틀린 게 아닙니다.** 서베이 제목을 쿼리로 주면 cross-encoder 는
"이 논문이 이 *주제에 관한* 것인가"를 묻습니다. 그러면 다른 서베이·개론 논문이 1위로
올라오고, 정작 서베이가 인용하는 **구체적인 방법·데이터셋 논문이 밀려납니다.**
정답은 "서베이가 인용한 논문"이므로 이 질문은 과녁을 벗어나 있습니다.

또 하나: RRF 점수에는 **facet 여러 개가 합의했다는 정보**가 들어 있는데, 쿼리 하나짜리
cross-encoder 는 그걸 통째로 버립니다.

> **다양성(MMR)에 이어 두 번째로, "당연히 좋을 것"이 정답 기준으로는 해로운 사례입니다.**
> 두 경우 모두 facet 이 만들어 낸 다중 쿼리 합의 신호를 파괴한다는 공통점이 있습니다.

**남은 선택지**: ⓐ `blend` 를 올려 RRF 신호를 일부 남기기(구현돼 있음),
ⓑ **논문을 끌어올린 facet 쿼리로 재랭킹**하기 — 논문마다 `facets` 를 갖고 있으므로
쌍 개수는 그대로이면서 쿼리가 구체적이 됩니다. ⓑ 가 원인을 정면으로 겨냥합니다.

### facet 쿼리 재랭킹 — 가설이 틀렸습니다 (같은 16토픽)

| 설정 | R@50 | R@500 | R@1500 | nDCG |
|---|---|---|---|---|
| facets (기준선) | **19.4%** | **45.1%** | **59.1%** | **0.288** |
| + rerank TOPIC | 14.0% | 36.6% | 56.3% | 0.190 |
| + rerank **FACET** | **10.9%** | **31.9%** | **51.3%** | **0.172** |
| + rerank FACET_MAX | 11.8% | 34.5% | 55.5% | 0.187 |

**facet 쿼리가 오히려 가장 나쁩니다.** "쿼리를 구체적으로 만들면 나아진다"는 가설이
틀렸습니다 — 정확히는, 구현이 **이 프로젝트가 이미 확립한 원칙을 어겼습니다.**

원인은 **쿼리 간 점수 비교 불가**입니다. 같은 논문 80편을 facet 쿼리별로 채점해 보면:

| facet 쿼리 | 평균 | 최대 |
|---|---|---|
| rationale extraction machine reading comprehension | 0.127 | 0.765 |
| faithfulness of explanations question answering | 0.076 | 0.961 |
| ERASER benchmark explainable NLP | 0.015 | 0.395 |
| chain-of-thought prompting reading comprehension | **0.006** | **0.037** |

쿼리별 평균이 **20배** 차이 납니다. 그래서 `chain-of-thought` facet 에서 **1위**인 논문이
`rationale extraction` facet 에서 **중위권**인 논문보다 낮은 점수를 받습니다. 결과적으로
"관련성"이 아니라 **"어느 facet 에 속했는가"로 정렬**됩니다.

이건 dense 점수에서 이미 겪은 것과 **같은 종류의 오류**입니다 — 이 저장소의 함정 목록
4번이 "쿼리 벡터가 비정규화라 쿼리 간 점수 비교 불가, 순위 기반 RRF 가 필수 조건"입니다.
cross-encoder 점수에도 똑같이 적용되는데 raw 점수를 직접 비교했습니다.
`FACET_MAX` 가 `FACET` 보다 나은 것도 이걸로 설명됩니다 — 3개 쿼리 중 최댓값을 쓰면
쿼리별 편차가 일부 상쇄됩니다.

**올바른 구현**: facet **안에서 순위를 매기고 facet 간에는 RRF 로 융합**. S4 와 같은 구조이고
`FACET_MAX` 와 쌍 개수가 같습니다. 다만 사전에 정한 중단 기준("기준선을 못 넘으면 접는다")에
따르면 여기서 멈추는 것이 맞고, 이 수정은 **가설 변경이 아니라 버그 수정**이라는 점만
구분해서 판단해야 합니다.

### 천장 — 검색 병목인가 랭킹 병목인가

정답을 못 맞히는 이유는 둘인데 처방이 정반대입니다. 후보 풀에 아예 없으면 **검색**
문제이고(표현·데이터를 손봐야 함), 풀에는 있는데 잘렸으면 **랭킹** 문제입니다
(재랭킹으로 해결, 본문 도입은 도움이 안 됨). 네 단계로 나눠 재면 갈라집니다.

토픽 16개, `facets+lexical+freshness`, 평균 후보 풀 12,134편:

| 단계 | recall | 무엇이 걸러졌나 |
|---|---|---|
| ① 코퍼스 상한 | 100.0% | 정답이 전부 우리 DB 에 있음 |
| ② 컷오프 상한 | 95.8% | 날짜 컷오프에서 4.2% 탈락 |
| ③ 풀 recall | **77.7%** | ← **검색 손실 18.1%p** |
| ④ 최종 recall | **59.1%** | ← **랭킹 손실 18.6%p** |

**검색과 랭킹이 거의 정확히 반반입니다.** 어느 한쪽만 고쳐서는 절반밖에 못 얻습니다.

- **랭킹 손실 18.6%p** — 후보 12,134편 중에는 있는데 상위 1,500편에 못 든 정답입니다.
  cross-encoder 재랭킹(P5.2)이 정확히 이 구간을 노립니다. **본문 없이 개선 가능한 몫**
- **검색 손실 18.1%p** — 후보에 아예 못 들어온 정답입니다. 여기가 표현(초록 vs 본문)이나
  임베딩 교체가 의미를 가질 수 있는 구간인데, **실제로는 대부분 깊이 부족이었습니다**(아래)

### 검색 손실은 표현이 아니라 깊이 부족이었다

"후보에 못 들어왔다"는 두 가지일 수 있고 처방이 정반대입니다 — 얕게 판 탓이면 `top_k`
한 줄이면 되고, 초록이 그 논문을 대표하지 못하는 것이면 본문이나 임베딩 교체가 필요합니다.
`top_k` 를 키워 가며 풀 recall 을 재면 갈라집니다 (토픽 8개):

| 쿼리당 `top_k` | 평균 후보 풀 | 풀 recall | 증분 |
|---|---|---|---|
| 500 | 3,933 | 73.3% | — |
| **2,000** (기본값) | 13,070 | 84.6% | +11.3%p |
| 8,000 | 42,686 | **91.1%** | +6.5%p |

**`top_k` 를 16배 키우니 recall 이 +17.8%p 올랐고, 컷오프 상한 95.8% 에 4.7%p 까지
접근했습니다.** 즉 초록으로 도달할 수 없는 몫은 18%p 가 아니라 **약 5%p** 입니다.

> **본문 도입은 지금 우선순위가 아닙니다.** 35GB·수 주가 드는 작업으로 얻을 수 있는
> 상한이 5%p 인데, `top_k` 설정 한 줄로 얻는 것이 17.8%p 입니다. 본문 논의는 깊이를
> 충분히 키우고 재랭킹을 붙인 뒤에 남는 격차를 보고 하는 것이 맞습니다.

> ⚠ 표본 16개입니다. ①②는 격차가 커서 뒤집히기 어렵지만, ③④는 표본을 늘려야
> 확정할 수 있습니다. recall 상한이 100% 가 아니라 **약 93%** 라는 점도 같이 보세요.

## 8. 구현 중 잡은 문제들

전부 "무음 폐기 금지" 원칙이 잡아냈습니다. `stats` 경고가 없었으면 조용히 넘어갔습니다.

| 문제 | 증상 | 조치 |
|---|---|---|
| 랭킹 윈도우가 후보를 굶김 | facet 켜면 후보 48,214편인데 S6·S7이 상위 3,000편만 봄 | 윈도우를 설정 가능하게, 기본 전량 |
| `provenance_of` 가 매번 집합 재구성 | 후보 48,000 × 원본 72,000 에서 **315초** | 집합을 호출부에서 한 번만 → 1.4초 |
| `get_papers` 의 `IN (?,?,…)` | id 3,000개에 19.7초 | pyarrow 조인 → 5,000개 0.32초 |
| 스노우볼링이 0편 기여 | S2는 버전 없는 id, 로컬 DB는 버전 붙은 id | `get_papers` 에 base_id 폴백 |
| 스노우볼링 논문이 컷에 못 듦 | 점수를 바닥에 깔아 경쟁 불가 | 임의 점수 대신 **RRF 융합** |

특히 마지막 둘은 기능이 "돌아가는 것처럼 보이면서" 아무 일도 안 하던 경우입니다.

## 9. 알아야 할 제약

**① 로컬 코퍼스는 2026-08-04 스냅샷입니다.** 그 이후 arXiv 는 존재하지 않는 것으로
취급됩니다. 랭킹 축의 최신성을 고쳐도 **데이터 축의 컷오프는 남습니다.** 이걸 메우려고
온라인 백엔드(arXiv API + S2)를 붙였습니다 — 실측하면 RAG 원논문의 피인용 309편 중
**76편(25%)** 이 로컬에 없습니다.

**② 원본 DB의 `date` 는 v1 게시일이 아니라 최신 버전 갱신일입니다.** v1 논문은 96%가
제출월과 일치하는데 v2+ 는 70%로 떨어집니다. 2007년 논문이 2025년에 개정된 사례도
있습니다. 그래서 arXiv id 의 `YYMM` 에서 제출월을 유도해 씁니다 — `date` 기준으로 재면
검색 결과의 최신성이 약 10%p 부풀려집니다.

**③ 방향 지표와 성능 지표를 구분하세요.** 최신성 비율·유사도·카테고리 수는 정답 없이
계산되므로 언제든 잴 수 있지만, **"얼마나 맞는가"는 말해 주지 않습니다.**

**④ 초록만 있습니다.** 본문 전체가 없어 구절 단위 검색·본문 근거 추출은 못 합니다.

## 10. 현재 상태

| 항목 | 상태 |
|---|---|
| **P0** 기반·인덱스 | ✅ id 매핑 왕복 10/10, DuckDB 908,819행 + BM25 |
| **P1** 최소 파이프라인 | ✅ dense + BM25 + RRF + dedup |
| **P2** facet · freshness · 다양성 | ✅ S1(OpenRouter) · S6 · S7 |
| **P3** 어댑터 · CLI | 🟡 시그니처·CLI 완료. **서베이 생성 스모크는 승인 대기** |
| **P4** 진단 하네스 | ✅ stats · 최신성 · 커버리지 · 베이스라인 대조 · 회귀 스냅샷 |
| **P5.1 / 5.7** 스노우볼링 · 온라인 | ✅ arXiv API + S2 인용 그래프, 디스크 캐시 |
| **P5.2** cross-encoder 재랭킹 | ✅ 구현 완료. **단, 정답 기준으로 기준선 미달** — §7 |
| **P5.5** SurGE 정량 평가 | ✅ 하네스 완료 + 토픽 16개 측정. 170개 중 나머지는 미실행 |

### 측정이 정한 권장 설정

```python
SearchConfig(n_papers=1500, facets=True, freshness=True)   # 나머지는 끕니다
```

`diversity` 와 `rerank` 는 구현돼 있지만 **정답 기준으로 기준선에 미달**했습니다(§7).
코드는 남겨 둡니다 — 커버리지가 진짜 목적 함수인 다른 설정에서는 결론이 다를 수 있고,
"무엇이 효과가 없었는가"도 이 프로젝트의 결과물이기 때문입니다.

내부 문서(`DESIGN.md` · `SETTING.md` · `TASKS.md` · `HANDOFF.md`)는 이 저장소에
포함되지 않습니다(`.gitignore`). 설계 근거·환경 실측·인수인계가 거기 있습니다.

## 11. 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 위치 | 독립 패키지 | 여러 에이전트가 공유. 베이스라인 레포를 오염시키지 않아야 통제 비교가 유지됨 |
| 1차 인덱스 | SurveyForge gte FAISS 재사용 + DuckDB BM25 신규 | GPU 재임베딩 0. `citation_count` 가 이미 있어 freshness 실험이 바로 가능 |
| 1차 범위 | 툴 + **규칙 기반** 오케스트레이터 | 결정적이라 단계별 ablation·디버깅이 쉬움 |
| 벡터 DB | FAISS + DuckDB (Milvus 아님) | 이 머신에 docker 소켓 권한이 없음 |
| 온라인 | 로컬을 **대체하지 않고 보강** | 로컬 = recall·결정성, 온라인 = 컷오프 이후 + 인용 엣지 |

## 12. SimScholarSearch 에서 가져온 것

파생 프로젝트지만 코드 수준으로 가져올 수 있는 건 많지 않습니다. 그쪽은 Milvus + BGE-M3 +
S2ORC 본문 + verl RL 스택이고, 우리는 FAISS + gte + arXiv 초록 + 규칙 기반이라 하부가
다릅니다. (라이선스 Apache-2.0, 출처는 각 파일 상단에 표기)

**가져온 것**

| 우리 쪽 | 원본 | 내용 |
|---|---|---|
| [`metrics/paper_set.py`](src/survey_search/metrics/paper_set.py) | `synthesis/paper_set.py`, `eval/litsearch.py` | `score_paper_set`(PFB adjusted_f1), LitSearch recall/nDCG. **id 타입만 int→str** |
| [`index/build_duckdb.py`](src/survey_search/index/build_duckdb.py) | `env/etl/build_fts.py`, `env/reader.py` | DuckDB FTS 사용법 |

패턴만 참고: `registry.py` 의 `build_registry()`/`compose()` 팩토리, `RRFRanker(60)`(k=60 선례),
툴 9종 시그니처(후속 ReAct 루프 착수 시).

**안 가져온 것** — `pymilvus` 의존 모듈 전부(Milvus 없음), `encoder.py`(BGE-M3는 별도 축),
`read_paper`/`find_in_paper`(본문 필요), `trainer`/`synthesis`/`agent`(verl RL 스택).

> 두 프로젝트의 id 체계가 다릅니다 — 그쪽은 S2ORC 정수 `corpus_id`, 이쪽은 버전 접미사가
> 붙은 arXiv 문자열 id. 이식한 코드에서 이 부분은 전부 바꿨습니다.

## 13. 참고한 레포

| 레포 | 이 프로젝트와의 관계 |
|---|---|
| [SimScholarSearch](https://github.com/trillion-labs/SimScholarSearch) | 이 프로젝트의 모태. 가져온 것과 안 가져온 것은 §12 |
| [AutoSurvey](https://github.com/AutoSurveys/AutoSurvey) | 1차 소비자. `database` 드롭인 어댑터 제공 |
| [SurveyForge](https://github.com/InternScience/SurveyForge) | 1차 소비자. `retrieve_id` 드롭인 어댑터 제공. 1차 인덱스도 여기서 재사용 |
| [SurGE](https://github.com/oneal2000/SurGE) | 평가 대상 (SIGIR 2026). GT 서베이 205편의 인용 목록이 정답 집합 |
