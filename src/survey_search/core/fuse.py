"""S4 — RRF 융합. DESIGN §S4.

    score(d) = Σ_q  1 / (k + rank_q(d)),   k = 60

**점수가 아니라 순위만 씁니다.** 이건 취향이 아니라 이 인덱스에서의 필수 조건입니다:

- dense 는 `IndexFlatIP` 인데 쿼리 벡터가 비정규화(norm ≈ 24)라 쿼리마다 점수
  스케일이 다릅니다 — 절대 점수를 더하면 norm 큰 쿼리가 결과를 지배합니다
- BM25 점수는 dense 와 애초에 단위가 다릅니다

2단으로 적용합니다: ① facet 내부에서 dense+BM25 융합 → ② facet 간 융합.
나누는 이유는 facet별 쿼터(S7)를 걸려면 facet 소속 정보가 살아 있어야 하기 때문입니다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from survey_search.backends.base import Hit

DEFAULT_K = 60


def rrf(
    ranked_lists: Sequence[Sequence[Hit]] | Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """여러 순위 목록을 RRF로 융합. 점수 내림차순으로 돌려줍니다.

    입력은 `(paper_id, score)` 목록이어도 되고 `paper_id` 목록이어도 됩니다 —
    어느 쪽이든 **점수는 무시하고 순서만** 씁니다.

    `weights` 로 목록별 가중을 줄 수 있습니다(예: dense 를 BM25 보다 무겁게).
    기본은 전부 1.0 — ablation 의 기준선을 단순하게 두기 위해서입니다.
    """
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(f"weights {len(weights)} != lists {len(ranked_lists)}")

    scores: dict[str, float] = {}
    for i, lst in enumerate(ranked_lists):
        w = 1.0 if weights is None else float(weights[i])
        if w == 0.0:
            continue
        for rank, item in enumerate(lst, start=1):
            pid = item[0] if isinstance(item, tuple) else item
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank)

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def provenance_of(
    paper_id: str, sources: dict[str, Iterable[Hit] | Iterable[str]]
) -> tuple[str, ...]:
    """어느 경로로 들어온 논문인지 — {"dense", "bm25", ...}.

    `stats` 만으로는 "BM25가 몇 편 새로 데려왔나"를 못 셉니다. 논문 단위로 남겨야
    "dense 가 못 잡은 걸 BM25 가 잡았다"는 주장을 논문 목록으로 보일 수 있습니다.
    """
    out = []
    for label, lst in sources.items():
        ids = {(x[0] if isinstance(x, tuple) else x) for x in lst}
        if paper_id in ids:
            out.append(label)
    return tuple(out)
