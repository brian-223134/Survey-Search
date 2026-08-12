# SETTING — 확인된 환경·자산 사실

**조사일**: 2026-08-12. 아래 수치는 전부 이 머신에서 직접 확인한 값입니다.
추정치는 "추정"이라고 명시했고, 확인이 필요한 항목은 §6에 모아 두었습니다.

---

## 1. 하드웨어

| 항목 | 값 |
|---|---|
| GPU | NVIDIA **L40S 46GB × 8** |
| 유휴 GPU | **2026-08-12 재확인: 5번을 뺀 7장 전부 유휴** (0·1·2·3·4·6·7 = 14~16 MiB). 5번만 37 GB / 79% 점유 |
| RAM | 503 GB (가용 448 GB) — 1.4 GB TinyDB 통째 적재에 여유 |
| CPU | 64 코어 |
| 디스크 | `/data2` 19T 중 **1.8T 여유** (91% 사용) |

> 앞선 조사(같은 날 오전)에서는 유휴 GPU가 1·6·7번뿐이었습니다. 점유 상황은 시간대에 따라
> 크게 바뀌므로 긴 작업 전에는 매번 `nvidia-smi`로 다시 확인하세요.

> 다른 사람의 작업이 대부분의 GPU를 쓰고 있습니다. 재임베딩처럼 긴 작업을 돌리기 전에
> `nvidia-smi`로 다시 확인하세요.

## 2. Milvus는 쓸 수 없습니다

`docker ps` → `permission denied ... /var/run/docker.sock`. 19530 포트도 닫혀 있습니다.
**docker 그룹 권한을 받기 전까지 Milvus는 선택지가 아닙니다.**

→ 벡터 인덱스는 **FAISS**, 메타데이터·BM25는 **DuckDB**로 갑니다.
(SimScholarSearch가 Milvus+DuckDB를 쓰지만, 그 DuckDB 파트만 차용하는 셈입니다.)

## 3. 논문 DB 자산

### 3-A. SurveyForge_data/database_2026-08 — **1차 인덱스로 채택**

경로: `/data2/chanjoong/survey-agent/SurveyForge_data/database_2026-08/`

| 파일 | 크기 | 내용 |
|---|---|---|
| `arxiv_paper_db_with_cc.json` | 1,400,883,616 B | TinyDB. **`citation_count` 포함** |
| `arxivid_to_index_abs.json` | 21,658,441 B | arxiv_id ↔ 인덱스 매핑 |
| `faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_260804.bin` | 3,729,793,266 B | title+abs 임베딩 |
| `faiss_paper_title_embeddings_FROM_2012_0101_TO_260804.bin` | 3,729,793,266 B | title 임베딩 |
| `faiss_survey_*` (2개) | 각 52,350,714 B | **서베이 전용** 인덱스 (1501~2409, gte) |

FAISS 헤더 직접 판독:

```
magic = IxMp (IndexIDMap)   d = 1024   ntotal = 908,819
```

- 임베딩 모델: **gte-large-en-v1.5** (1024d)
- 수록 범위: 1991-03-13 ~ 2026-08-04 (SurveyForge `HANDOFF.md`)
- md5 지문은 `../SurveyForge/REPRODUCTION.md` §7.4에 있습니다

> ✅ **id 매핑은 2026-08-12에 실측으로 해소됐습니다 — §6-A 참조. 이중 매핑이 아니라
> 1-based 단일 매핑이고, 의미 수준 왕복 검증 10/10 통과했습니다.**

### 3-B. AutoSurvey/database_2026-08 — 2차 백엔드

경로: `/data2/chanjoong/survey-agent/AutoSurvey/database_2026-08/`

```
magic = IxF2 (IndexFlatL2)   d = 768   ntotal = 909,293
```

- 임베딩 모델: **nomic-embed-text-v1** (768d, `search_query:` / `search_document:` prefix 사용)
- 레코드 필드: `id`, `title`, `url`, `date`, `abs`, `cat`, `authors` — **`citation_count` 없음**
- 수록 범위: ~2026-08-03. 배포본 537,665편의 상위 집합 (배포본이 prefix)

### 3-C. 두 DB의 관계

같은 arXiv를 각자 수집한 **별개 스냅샷**입니다(908,819 vs 909,293). id 체계는 둘 다
버전 접미사가 붙은 arXiv id(`1811.06122v1`)라, 교차 정합의 키는 **버전을 뗀 base id**입니다.

## 4. 임베딩 모델 캐시

`/data2/chanjoong/.cache/huggingface/hub/` 에 이미 받아져 있습니다:

| 모델 | 용도 |
|---|---|
| `Alibaba-NLP/gte-large-en-v1.5` | **1차 인덱스의 쿼리 임베딩** (필수) |
| `Alibaba-NLP/new-impl` | gte의 trust_remote_code 의존 |
| `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-bert-2048` | AutoSurvey 백엔드용 |
| `BAAI/bge-large-en-v1.5` | 예비 |
| `cross-encoder/nli-deberta-v3-base` | 재랭킹용은 **아님**(NLI 모델). 재랭커는 별도 다운로드 필요 |

**없는 것**: `BAAI/bge-m3` (후속 하이브리드 인덱스용), `BAAI/bge-reranker-v2-m3` (cross-encoder 재랭킹용).

## 5. 파이썬 환경

| 항목 | 값 |
|---|---|
| 시스템 python3 | 3.10.12 — **faiss 없음** |
| 기존 venv | `../SurveyForge/.venv` (faiss·sentence-transformers·langchain 포함, 3.10) |
| conda | `/data2/chanjoong/miniforge3` |
| uv | 없음 |

survey-search는 **자체 venv**를 만드는 것을 권장합니다. SurveyForge의 venv는
langchain 계열이 무겁게 얽혀 있어 그대로 상속하면 의존성이 오염됩니다.

## 6. 실측 결과 (2026-08-12 측정 완료)

A~E는 전부 실측했습니다. 아래는 추정이 아니라 이 머신에서 나온 값입니다.
측정 스크립트는 P0.2에서 `index/inspect_faiss.py`로 정식화할 예정입니다.

### A. FAISS id 매핑 — ✅ 해소, 이중 매핑 아님

```
IndexIDMap.id_map        = {1..908819} 의 순열 (1-based, 전단사)
                           단, 행 순서와 일치하지 않음 — 441,842개가 어긋남
arxivid_to_index_abs.json = 1 ~ 908819, unique 908,819개 — id_map 집합과 일치
내부 인덱스               = IndexFlatIP   (L2가 아니라 내적)
```

> **정정** (2026-08-12 재측정): 처음에 "1-based 연속"이라고 적었으나 **"연속"은 틀렸습니다.**
> id_map 은 1..N 의 순열이되 오름차순이 아닙니다. 행 147,281부터 id가 441,844로 점프하고
> 뒤에서 되돌아옵니다(diff 값이 `{-294561, -294560, 1, 294562, 294563}` 5종) — 인덱스가
> **세 덩어리를 다른 순서로 이어 붙여** 만들어졌다는 흔적입니다.

`index.search()`가 돌려주는 id가 곧 json의 값이므로 **`index_to_id`를 한 번 적용하는 것이 정답**입니다.
SurveyForge 코드([database.py:63](../SurveyForge/code/src/database.py#L63))가 맞게 하고 있습니다.

**의미 수준 왕복 검증 10/10 통과** — DB에서 논문 10편을 뽑아 title+abs를 gte로 임베딩 →
검색 → top-1의 id를 `index_to_id`로 되돌린 결과가 원래 arxiv_id와 정확히 일치
(9만 간격 샘플: `1811.06122v1`, `2402.08314v1`, … `2604.16172v1`).

> ⚠ **0-based로 가정하면 조용히 한 칸 밀린 논문이 반환됩니다.** 새 백엔드 구현에서
> 반드시 1-based를 유지하세요. 왕복 검증은 `tests/test_faiss_mapping.py` 로 고정해 뒀습니다.

> ⚠ **`faiss_id - 1` 은 행 번호가 아닙니다.** 순열이라서 그렇습니다. 저장된 문서 벡터를
> 꺼내야 하는 곳(S7 MMR의 초록 임베딩 유사도)에서는 반드시
> `index/inspect_faiss.py` 의 `build_id_to_row()` / `reconstruct_by_id()` 를 거치세요.
> id 로만 접근하는 일반 검색 경로는 이 문제와 무관합니다.

> ⚠ **정규화 비대칭** — 저장된 문서 벡터는 단위 norm인데 gte가 내놓는 쿼리 벡터는 norm ≈ 24로
> 정규화돼 있지 않습니다(자기 자신과의 IP ≈ 23.9). 한 쿼리 안에서의 **순위는 영향받지 않지만
> 쿼리끼리 점수를 비교할 수는 없습니다.** DESIGN이 점수 대신 순위 기반 RRF를 택한 것이
> 이 인덱스에서는 선택이 아니라 필수 조건입니다.

### B. 검색 지연 — ✅ 충분히 빠름, HNSW 불필요

| 측정 | 값 |
|---|---|
| 인덱스 로드 (3.7 GB) | **2.3 ~ 5.0 s** |
| **첫 검색 (cold)** | **12.2 s** — 3.7GB를 실제로 읽어 들이는 페이지 폴트 비용 |
| 1 쿼리 × top-1500 (warm) | **0.08 ~ 0.79 s** |
| 32 쿼리 배치 × top-1500 (warm) | **2.73 s 총 (85 ms/쿼리)** |

> ⚠ **cold/warm 차이가 150배입니다.** 프로세스를 새로 띄울 때마다 첫 쿼리에 12초를
> 냅니다. 배치 실험에서는 백엔드를 재사용하고, 지연을 잴 때는 반드시 워밍업 후에 재세요.

`OMP_NUM_THREADS=16`, CPU only. facet 16개 × 인덱스 2개 = 32 쿼리가 **3초 미만**입니다.
→ **HNSW 변환도 faiss-gpu도 필요 없습니다** (TASKS의 P5.4는 착수 사유가 사라졌습니다).
배치가 쿼리당 9배 빠르므로 Backend 프로토콜의 배치 API 설계가 실측으로 정당화됩니다.

### C. `citation_count` 분포 — ✅ 결측 0, 단 **문자열 타입**

| 항목 | 값 |
|---|---|
| 전체 | 908,819 |
| 결측(None/필드 없음) | **0** |
| `== 0` | 251,958 (27.7%) |
| `> 0` | 656,861 (72.3%) |
| `> 0`의 중앙값 | **6** |
| 최댓값 | 175,503 |

> ⚠ 값이 `'17'`처럼 **문자열로 저장**돼 있습니다. DuckDB 적재 시 `CAST(... AS BIGINT)` 필수이고,
> 문자열 정렬로 두면 `'9' > '175503'`이 되어 랭킹이 조용히 망가집니다.

인용수 중앙값이 6에 불과하다는 점이 중요합니다. 상위권이 소수의 초고인용 논문에 지배되므로
S6의 **연령 정규화 백분위** 설계가 절대 인용수 정렬보다 유리할 여지가 큽니다.

### D. `date` 분포 — ✅ 전 레코드 존재, ISO 날짜

연도별 건수 (최근 8년):

| 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| 52,813 | 66,491 | 70,522 | 76,175 | 92,979 | 111,569 | **139,160** | **138,644** |

**2025~2026년 논문이 277,804편, 전체의 30.6%입니다.** 2026년은 8월까지 7개월 만에 138,644편으로
이미 2025년 전체에 육박합니다. 즉 **코퍼스의 3할이 인용수 정렬에서 구조적으로 탈락하는 구간**이며,
이 프로젝트의 전제가 데이터로 확인됩니다.

> 남은 확인: `date`가 v1 최초 게시일인지 최신 버전 갱신일인지는 아직 미확정입니다.
> id에 `v3` 같은 접미사가 붙은 레코드의 date를 arXiv API와 대조하면 확정됩니다 (P0.6 잔여분).

### E. TinyDB JSON 적재 — ✅ 추정보다 훨씬 빠름

| 항목 | 값 |
|---|---|
| `json.load()` 1.4 GB 전체 파싱 | **14~23 s** (추정 10~30분은 과대평가였음) |
| 테이블 | `cs_paper_info` 단일, 908,819 행 |
| 레코드 필드 | `id`, `title`, `url`, `date`, `abs`, `cat`, `authors`, `citation_count` |
| 고유 `id` 수 | 908,819 — **전단사 확인** |

파싱이 병목이 아니므로 P0.4의 DuckDB 적재는 단순 일괄 INSERT로 충분합니다.

## 7. 아직 없는 의존성

| 패키지 | 상태 |
|---|---|
| `faiss` 1.9.0, `sentence_transformers` 2.7.0 | ✅ SurveyForge venv에 있음 (python 3.10.12) |
| `duckdb` | ✅ 자체 venv에 1.5.5 설치 완료 |
| `rank_bm25` | 불필요 — DuckDB FTS 로 갑니다 |

### venv 만들 때 걸린 것 두 가지

- **`python3 -m venv` 가 안 됩니다** — `ensurepip` 없음(`apt install python3.10-venv` 권한 없음).
  `virtualenv -p python3.10 .venv` 로 우회합니다. conda(`/data2/chanjoong/miniforge3`)도 가능
- **torch 최신판은 이 머신에서 GPU를 못 씁니다** — 드라이버가 550.120(CUDA 12.4)인데
  torch 2.13 기본 휠은 더 최신 드라이버를 요구합니다(`driver ... too old (found version 12040)`).
  **cu124 빌드로 고정하세요**:
  `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` → GPU 8장 인식 확인

LLM 키는 `../AutoSurvey/.env`, `../SurveyForge/.env`, `../SurGE/.env`에 있습니다 (facet 분해용, DESIGN §7-3).
