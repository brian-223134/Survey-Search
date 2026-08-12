# survey-search — 에이전트용 안내

토픽 하나를 받아 관련 논문 집합을 찾는 **검색 레이어**. 여러 서베이 에이전트
(AutoSurvey · SurveyForge · SurveyX · 향후 것들)가 공유해 쓰는 독립 패키지입니다.

**현재 상태 (2026-08-12): 설계 문서 + P0 실측 검증 완료. 코드는 아직 없습니다.**
기존 FAISS 인덱스가 쓸 수 있다는 것은 확인됐습니다(id 매핑 왕복 10/10, 32쿼리 2.73초).
**P0.1 스캐폴딩부터 시작하세요.**

## 먼저 읽을 것 — 이 순서로

| 문서 | 왜 |
|---|---|
| [`README.md`](README.md) | 왜 만드는지, 확정된 결정 4개 |
| [`SETTING.md`](SETTING.md) | 이 머신의 확인된 사실. **§6은 구현 전 실측해야 할 항목** |
| [`DESIGN.md`](DESIGN.md) | Backend 프로토콜, 파이프라인 8단계 사양, 어댑터 규약 |
| [`TASKS.md`](TASKS.md) | P0~P5. **다음 한 걸음은 P0.1 → P0.4/0.5** |

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
- **유휴 GPU는 1·6·7번**. 긴 작업 전에 `nvidia-smi`로 다시 확인
- **서베이 생성은 사용자 승인 후에만** — 편당 실비 $0.3~2. 검증·집계처럼 돈이 안 드는 구간은 알아서 진행
- **형제 레포는 읽기 전용** — `../AutoSurvey`, `../SurveyForge`, `../SurveyForge_data`,
  `../SimScholarSearch`를 고치지 마세요. 통제 비교가 깨집니다

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
../SurGE/                               후속 평가 대상 (구현 코드 미공개)
```
