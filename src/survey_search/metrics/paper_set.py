"""정답 집합이 있을 때의 검색 채점기.

SimScholarSearch에서 이식했습니다 (Apache-2.0, trillion-labs/scholar-search-rl):

- `score_paper_set` / `_dcg` / `_ndcg_rank` / `_harmonic`
  ← `src/s2cs/synthesis/paper_set.py` (PaperFindingBench 의 adjusted_f1)
- `calculate_recall` / `calculate_ndcg`
  ← `src/s2cs/eval/litsearch.py` (princeton-nlp/LitSearch 원본을 그대로 옮긴 것)

바꾼 것은 **id 타입뿐**입니다. 그쪽은 S2ORC 정수 `corpus_id`, 이쪽은 arXiv 문자열
id(`"2401.12345v2"`)입니다. 버전 접미사 때문에 같은 논문이 다른 id로 보일 수 있으므로,
채점 전에 `base_id` 로 정규화해서 넣으세요 (`normalize_ids` 참고).

지금 당장은 쓸 정답 집합이 없습니다(SurGE 미공개). 그래도 먼저 옮겨 두는 이유는,
벤치마크가 열렸을 때 채점기부터 새로 짜면 그 시점의 구현이 곧 결과를 좌우하기
때문입니다. 지표는 실험보다 먼저 고정돼 있어야 합니다.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

_VERSION_SUFFIX = re.compile(r"v\d+$")


def normalize_ids(ids: Iterable[str]) -> list[str]:
    """버전 접미사를 떼고 순서를 유지한 채 중복 제거.

    `2401.12345v1` 과 `v2` 를 같은 논문으로 채점하기 위한 전처리입니다.
    """
    out: list[str] = []
    seen: set[str] = set()
    for pid in ids:
        base = _VERSION_SUFFIX.sub("", str(pid))
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


# --- LitSearch 표준 지표 (순서 무관 recall + 순위 반영 nDCG) --------------------

def calculate_recall(retrieved: list[str], relevant_docs: list[str]) -> float:
    """princeton-nlp/LitSearch `utils/utils.py` 원본 그대로."""
    num_relevant_retrieved = len(set(retrieved).intersection(set(relevant_docs)))
    num_relevant = len(relevant_docs)
    return num_relevant_retrieved / num_relevant if num_relevant > 0 else 0.0


def calculate_ndcg(retrieved: list[str], relevant_docs: list[str]) -> float:
    """princeton-nlp/LitSearch `utils/utils.py` 원본 그대로.

    주의: 표준 nDCG의 `1/log2(i+1)` 이 아니라 `1/(i+1)` 할인을 씁니다.
    LitSearch 원본이 그렇게 돼 있어서 비교 가능성을 위해 유지합니다.
    """
    dcg = 0.0
    for idx, docid in enumerate(retrieved):
        if docid in relevant_docs:
            dcg += 1 / (idx + 1)
    idcg = sum(1 / (idx + 1) for idx in range(len(relevant_docs)))
    return dcg / idcg if idcg > 0 else 0.0


# --- PaperFindingBench adjusted_f1 = harmonic(recall@est, nDCG-rank) ----------

def _dcg(grades: list[int]) -> float:
    # PFB find_dcg: 1위 -> /log(2), 2위 -> /log(3), ...
    return sum(g / math.log(i + 1) for i, g in enumerate(grades, start=1))


def _ndcg_rank(grades: list[int]) -> float:
    """제출 순서의 nDCG를 이상/최악 순서로 정규화 (PFB lower_bound_corrected_ndcg).

    원본과 다른 점 하나: 모든 등급이 같아 hi == lo 인 경우 PFB는 0을 주는데,
    여기서는 1.0을 줍니다. 전부 정답인 제출이 0점이 되는 건 지표가 틀린 것입니다.
    """
    if not grades:
        return 0.0
    hi, lo = _dcg(sorted(grades, reverse=True)), _dcg(sorted(grades))
    if hi == lo:
        return 1.0 if hi > 0 else 0.0
    return (_dcg(grades) - lo) / (hi - lo)


def _harmonic(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if (a > 0 and b > 0) else 0.0


def score_paper_set(
    predicted_ids: list[str],
    relevance: Mapping[str, int] | Iterable[tuple[str, int]],
    *,
    est_total_relevant: int | None = None,
    rel_threshold: int = 3,
    normalize: bool = True,
) -> dict:
    """**순서가 있는** 예측 집합을 등급 매겨진 정답과 대조합니다.

    `relevance` 는 paper_id -> 등급(1~3). 제출되지 않았거나 모르는 논문은 0점.
    `reward` 는 다음 둘의 조화평균입니다:

    - **recall@est**: 제출 순서 상위 `est` 개 중 "관련" 논문의 비율.
      PFB 정의상 관련 = **등급 3(PERFECT)만** (`rel_threshold=3`). 등급 1~2는
      recall 에 안 들어갑니다. `est` 로 자르는 것이 패딩 방지 역할을 하며,
      precision 항은 없습니다(broad 질의 지표라 그렇습니다).
    - **rank**: 제출된 전체 등급(0~3)의 순서에 대한 nDCG. 등급 1~2가 여기서
      "더 관련 있는 것을 앞에" 라는 신호로 쓰입니다.

    `normalize=True` 면 예측·정답 id 를 모두 버전 없는 base_id 로 맞춥니다.
    """
    rel_pairs = relevance.items() if isinstance(relevance, Mapping) else relevance
    rel = {str(c): int(g) for c, g in rel_pairs}
    preds = [str(p) for p in predicted_ids]

    if normalize:
        rel = {_VERSION_SUFFIX.sub("", k): v for k, v in rel.items()}
        preds = normalize_ids(preds)
    else:
        preds = list(dict.fromkeys(preds))

    total = (
        est_total_relevant
        if est_total_relevant is not None
        else sum(1 for g in rel.values() if g >= rel_threshold)
    )

    relevant_at_est = sum(1 for p in preds[:total] if rel.get(p, 0) >= rel_threshold)
    recall = relevant_at_est / total if total else 0.0
    rank = _ndcg_rank([rel.get(p, 0) for p in preds])

    return {
        "reward": _harmonic(recall, rank),
        "recall_at_est": recall,
        "rank": rank,
        "relevant_at_est": relevant_at_est,
        "n_pred": len(preds),
        "n_gold": total,
    }
