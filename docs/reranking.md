# cross-encoder 재랭킹 — 세 변형 전부 기준선 미달

**한 줄**: 재랭커가 틀린 게 아니라 **쿼리와 점수 설계가 틀렸습니다.** 그리고 그 오류는
이 저장소가 이미 문서화해 둔 함정이었습니다.

**결론**: 기본값 `rerank=False`. 재시도는 열린 결정으로 남아 있습니다.

## 왜 켜볼 만했나

천장 측정에서 **랭킹 손실이 18.6%p** 로 검색 손실(18.1%p)과 맞먹었습니다
→ [ceiling-and-depth.md](ceiling-and-depth.md). 후보 풀에는 정답이 있는데 상위
1,500편 안에 못 넣고 있다는 뜻입니다.

`top_k` 를 4배 키워도 최종 recall 이 +1.1%p 밖에 안 올랐던 것도 같은 방향을 가리켰습니다
— **풀을 키우는 것으로는 안 되고 랭킹을 고쳐야 한다.**

cross-encoder(`BAAI/bge-reranker-v2-m3`)는 쿼리와 문서를 함께 보고 점수를 매기므로,
bi-encoder 검색보다 정밀합니다. 자연스러운 다음 수였습니다.

## 시도 1 — 토픽을 쿼리로 (TOPIC)

| 설정 | R@500 | R@1500 | nDCG |
|---|---|---|---|
| facets (기준선) | **45.1%** | **59.1%** | **0.288** |
| + rerank TOPIC | 36.6% | 56.3% | **0.190** |
| + rerank TOPIC + `top_k=8000` | 36.6% | 55.4% | 0.192 |

**nDCG 가 34% 떨어집니다.** 한 토픽을 열어 보니 이유가 분명했습니다.

서베이 제목("Explainable NLP 서베이")을 쿼리로 주면 재랭커는 **"이 주제에 관한 논문"**
을 위로 올립니다 — 다른 서베이, 개론, 포지션 페이퍼. 그런데 서베이가 실제로 인용하는
것은 **구체적인 방법·데이터셋 논문**입니다.

**HotpotQA 가 7위에서 940위로 떨어졌습니다.** 재랭커는 제 일을 정확히 했고, 우리가
잘못된 질문을 준 것입니다.

## 시도 2 — facet 쿼리를 쓰면 나아질 것이다 (FACET / FACET_MAX)

가설: 쿼리가 너무 추상적이었다. 그 논문을 끌어올린 **facet 쿼리**로 재랭킹하면
구체적이 되니 나아질 것이다. 쌍 개수도 그대로다.

| 설정 | R@50 | R@500 | R@1500 | nDCG |
|---|---|---|---|---|
| facets (기준선) | **19.4%** | **45.1%** | **59.1%** | **0.288** |
| + rerank TOPIC | 14.0% | 36.6% | 56.3% | 0.190 |
| + rerank **FACET** | **10.9%** | **31.9%** | **51.3%** | **0.172** |
| + rerank FACET_MAX | 11.8% | 34.5% | 55.5% | 0.187 |

**facet 쿼리가 오히려 가장 나쁩니다.** 가설이 틀렸습니다.

## 원인 — 이미 알고 있던 함정을 다시 밟았습니다

같은 논문 80편을 facet 쿼리별로 채점해 봤습니다:

| facet 쿼리 | 평균 | 최대 |
|---|---|---|
| rationale extraction machine reading comprehension | 0.127 | 0.765 |
| faithfulness of explanations question answering | 0.076 | 0.961 |
| ERASER benchmark explainable NLP | 0.015 | 0.395 |
| chain-of-thought prompting reading comprehension | **0.006** | **0.037** |

**쿼리별 평균이 20배 차이 납니다.**

그래서 `chain-of-thought` facet 에서 **1위**인 논문이 `rationale extraction` facet 의
**중위권** 논문보다 낮은 점수를 받습니다. raw 점수로 정렬하면 결과적으로
**"관련성"이 아니라 "어느 facet 에 속했는가"로 정렬**됩니다.

이건 이 저장소의 **함정 4번과 같은 오류**입니다:

> 쿼리 벡터가 비정규화라 쿼리 간 점수 비교 불가. 순위 기반 RRF 가 필수 조건.

dense 점수에 대해 이미 겪고 문서화까지 해뒀는데, cross-encoder 점수에 같은 규칙이
적용된다는 것을 놓쳤습니다. → [pitfalls.md](pitfalls.md) 함정 8

`FACET_MAX` 가 `FACET` 보다 나은 것도 이걸로 설명됩니다 — 3개 쿼리 중 최댓값을 쓰면
쿼리별 편차가 일부 상쇄됩니다.

## 고친다면 — facet 내 순위 + RRF

**올바른 구현**: facet **안에서 순위를 매기고 facet 간에는 RRF 로 융합.**
S4 와 같은 구조이고 `FACET_MAX` 와 쌍 개수가 같습니다.

고칠 위치는 `core/rerank.py` 의 `CrossEncoderReranker.rerank()` — raw 점수를 그대로
정렬 키로 쓰는 부분을 facet 별 순위로 바꾸고, `core/fuse.py` 의 `rrf()` 로 합칩니다.

## 열린 결정

사전에 정한 중단 기준이 있습니다 — **"기준선을 못 넘으면 접는다."**

세 번 시도해 셋 다 미달했으니 기준대로는 여기서 멈추는 것이 맞습니다.
다만 이 수정은 **가설 변경이 아니라 버그 수정**이라는 점은 구분해야 합니다.
"재랭킹이 도움이 되는가"라는 질문에 아직 제대로 답한 적이 없기 때문입니다.

**아직 결정되지 않았습니다.**

## 재현

```bash
python scripts/eval_rerank.py          # TOPIC (약 60분)
python scripts/eval_rerank_facet.py    # FACET / FACET_MAX (약 45분)
```

> ⚠ 위 수치는 전부 **토픽 16개** 기준입니다. 기준선이 R@1500 59.1% / nDCG 0.288 이고,
> 170토픽의 57.5% / 0.299 가 아닙니다. **비교는 같은 표본끼리만 하세요.**

## 관련 문서

- [ceiling-and-depth.md](ceiling-and-depth.md) — 왜 랭킹을 고치려 했나
- [diversity-mmr.md](diversity-mmr.md) — 같은 원인으로 실패한 다른 단계
- [pitfalls.md](pitfalls.md) — 함정 4·8
