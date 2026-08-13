# survey-search — 에이전트용 안내

토픽 하나를 받아 관련 논문 집합을 찾는 **검색 레이어**. 여러 서베이 에이전트
(AutoSurvey · SurveyForge · SurveyX · 향후 것들)가 공유해 쓰는 독립 패키지입니다.

**현재 상태 (2026-08-13): P0~P5 구현·측정 완료, 테스트 154개 통과.**
파이프라인 전 단계(dense+BM25+RRF+dedup+freshness+MMR+facet+스노우볼링+재랭킹)와
온라인 백엔드(arXiv·S2), SurGE 정답 기준 평가까지 끝났습니다.

**측정이 말하는 결론 한 줄: `facets=True, freshness=True` 가 최선이고, 다양성(MMR)과
cross-encoder 재랭킹은 켜면 손해입니다.** 셋 다 "당연히 좋을 것"이었는데 정답 기준으로는
기준선(R@1500 59.1%, nDCG 0.288)에 미달했고, 공통 원인은 **facet 이 만든 다중 쿼리
합의(RRF) 신호를 파괴**한다는 것입니다.

**다음 판단이 필요한 것** — 재랭킹을 facet 내 순위 + RRF 융합으로 한 번 더 시도할지,
아니면 사전에 정한 중단 기준대로 접을지. 자세한 것은 `HANDOFF.md` §8.

⚠ `SETTING.md` · `TASKS.md` · `DESIGN.md` · `HANDOFF.md` 는 `.gitignore` 처리돼 **커밋되지 않습니다.**
로컬에는 그대로 있으니 계속 읽고 갱신하세요.

## 먼저 읽을 것 — 이 순서로

| 문서 | 왜 |
|---|---|
| `HANDOFF.md` | **인수인계 — 여기부터 읽으세요.** 실측 사실·함정·다음 할 일 (**git 미추적**) |
| [`README.md`](README.md) | 왜 만드는지, 구조·원리, 결과표 (공개용) |
| `SETTING.md` | 이 머신의 확인된 사실. §6에 P0 실측 결과 (**git 미추적 — 로컬 전용**) |
| `DESIGN.md` | Backend 프로토콜, 파이프라인 8단계 사양, 어댑터 규약 (**git 미추적**) |
| `TASKS.md` | P0~P5과 전체 실측 결과표 (**git 미추적**) |

## 이 프로젝트의 요지 한 문단

기존 서베이 에이전트의 검색은 전부 "dense 벡터 1방 + top-k"입니다. DB를 2026-08로 최신화해도
**랭킹이 최신 논문을 구조적으로 배제**합니다 — SurveyForge의 `sort_by_citation_period`는
인용수 정렬이라 최근 6~12개월 논문은 인용수가 0에 가까워 다시 탈락합니다. 이 패키지는
멀티쿼리 fan-out · BM25 하이브리드 · freshness-aware 랭킹 · 다양성 제어를 **끄고 켤 수 있는
단계로** 쌓아, 각 단계의 기여를 분리 측정합니다.

## 지켜야 할 제약

- **Milvus 못 씁니다** — docker 소켓 권한 없음. FAISS + DuckDB로 갑니다
- **재임베딩 하지 않습니다** — 1차는 SurveyForge의 기존 gte 인덱스(1024d, 908,819편) 재사용.
  BGE-M3 재구축은 P5의 별도 축
- **GPU는 여러 사람이 공유합니다.** 유휴 카드는 수시로 바뀌고, 8장이 전부 찬 시점도
  실제로 있었습니다. 긴 작업 전에 `nvidia-smi`로 확인하고 `CUDA_VISIBLE_DEVICES`로
  빈 카드를 명시하세요 — 지정 안 하면 0번에 겹쳐 올라가 OOM이 납니다
- **서베이 생성은 사용자 승인 후에만** — 편당 실비 $0.3~2. 검증·집계처럼 돈이 안 드는 구간은 알아서 진행
- **형제 레포는 읽기 전용** — `../AutoSurvey`, `../SurveyForge`, `../SurveyForge_data`,
  `../SimScholarSearch`를 고치지 마세요. 통제 비교가 깨집니다
- **GIT은 오직 Commit 까지만 진행** - Commit을 진행할 때, { docs, feat, modify ... }와 같이 message를 쓰기 (ex: docs: README에 실험 결과 추가)

## 작업 규칙

- **각 작업은 검증을 통과해야 다음으로 넘어갑니다.** 검증은 "돌려보고 숫자를 확인한다"이며,
  코드를 눈으로 훑는 것은 검증이 아닙니다
- **무음 폐기 금지** — 필터·윈도우·컷오프가 버린 논문 수는 반드시 세어서 `stats`에 남깁니다.
  SurveyForge에서 날짜 윈도우가 예외도 로그도 없이 논문을 버리던 문제가 실제로 있었습니다
- 진행 상황이 바뀌면 `TASKS.md`의 표와 `README.md`의 현재 상태를 갱신하세요
- 실측한 값(지연 시간, 분포, 건수)은 `SETTING.md`에 기록하세요. 추정치와 실측치를 구분해서

## 관련 경로

```
../SurveyForge_data/database_2026-08/   1차 인덱스 (gte 1024d, citation_count 포함)
../AutoSurvey/database_2026-08/         2차 백엔드 (nomic 768d)
../SurveyForge/, ../AutoSurvey/         1차 소비자. 각 HANDOFF.md에 DB 최신화 경위
../SimScholarSearch/                    툴 인터페이스·paper_set 채점기 참고용
../SurGE/                               평가 대상. data/surveys.json 의 205편 ×
                                        all_cites 가 정답 집합 (queries.json 은 없어도 됨)
```
