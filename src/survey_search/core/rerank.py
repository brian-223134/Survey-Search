"""S8b — cross-encoder 재랭킹. DESIGN §S8.

**왜 이 단계인가.** 천장 측정에서 랭킹 손실이 18.6%p 로 나왔습니다 — 정답이 후보 풀
안에는 들어와 있는데 최종 컷 아래로 밀린 몫입니다. 후보에는 이미 있으므로 **본문도
재임베딩도 필요 없고**, 순서만 고치면 되찾을 수 있는 구간입니다.

bi-encoder(gte)는 쿼리와 문서를 **따로** 인코딩합니다 — 그래야 90만 편을 미리 색인할 수
있지만, 쿼리와 문서가 서로를 보지 못합니다. cross-encoder 는 둘을 **함께** 넣어 한 번에
점수를 냅니다. 정확하지만 후보 수만큼 forward 를 돌려야 해서 색인에는 못 쓰고,
**상위 수백~수천 편을 다시 세우는 데만** 씁니다.

비용이 실질적입니다. 후보 1,000편 × 512토큰이면 GPU 에서 수 초, CPU 에서는 분 단위입니다.
그래서 기본적으로 `top_n` 만 재랭킹하고 나머지는 원래 순서를 유지합니다 — 잘라낸 건수는
`stats` 에 남습니다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from survey_search.types import Paper

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankConfig:
    model: str = DEFAULT_MODEL
    #: 상위 몇 편을 재랭킹할지. 나머지는 원래 순서 뒤에 그대로 붙습니다.
    #: **`n_papers` 보다 커야 의미가 있습니다.** 작으면 최종 집합이 안 바뀌고 순서만
    #: 바뀌어서, 재랭킹을 켠 실험과 끈 실험이 recall 로 구분되지 않습니다.
    #: 기본값 3000 은 `n_papers=1500` 기준으로 2배입니다.
    top_n: int = 3000
    batch_size: int = 64
    max_length: int = 512
    device: str = "auto"
    #: 문서 텍스트를 만들 때 초록을 몇 자까지 쓸지. 512토큰 상한을 넘기면 잘리므로
    #: 제목이 살아남도록 초록을 먼저 자릅니다.
    abstract_chars: int = 1200
    #: 최종 점수 = (1-blend)·cross_score_norm + blend·original_rank_score.
    #: 0.0 이면 cross-encoder 만, 1.0 이면 재랭킹 안 한 것과 같습니다.
    #: 0 이 아닌 값을 쓰는 이유: cross-encoder 는 쿼리 하나만 보므로 facet 여러 개가
    #: 합의해 올린 논문의 정보를 잃습니다. 그 신호를 일부 남깁니다.
    blend: float = 0.0


@dataclass
class RerankStats:
    applied: bool = False
    n_in: int = 0
    n_scored: int = 0
    n_untouched: int = 0        # top_n 밖이라 원래 순서를 유지한 수
    elapsed_s: float = 0.0
    device: str = ""
    model: str = ""
    note: str = ""
    errors: list[str] = field(default_factory=list)


def _resolve_device(want: str) -> str:
    if want != "auto":
        return want
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _doc_text(p: Paper, abstract_chars: int) -> str:
    """cross-encoder 에 넣을 문서 표현. 제목이 잘리지 않도록 초록을 먼저 자릅니다."""
    abstract = (p.abstract or "").strip()[:abstract_chars]
    return f"{p.title}\n{abstract}".strip()


class CrossEncoderReranker:
    """모델을 한 번만 로드해 재사용합니다. 매 검색마다 2.2GB 를 다시 읽으면 안 됩니다."""

    def __init__(self, config: RerankConfig | None = None) -> None:
        self.config = config or RerankConfig()
        self._model = None
        self._device: str | None = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = _resolve_device(self.config.device)
            t = time.perf_counter()
            try:
                m = CrossEncoder(self.config.model, max_length=self.config.max_length,
                                 device=device)
            except RuntimeError as e:
                # GPU 를 여러 사람이 공유합니다 — 못 올리면 CPU 로 물러섭니다.
                if device == "cpu":
                    raise
                log.warning("%s 로 올리지 못해 CPU 로 폴백합니다: %s", device, e)
                device = "cpu"
                m = CrossEncoder(self.config.model, max_length=self.config.max_length,
                                 device=device)
            log.info("cross-encoder loaded in %.1fs (%s)", time.perf_counter() - t, device)
            self._model, self._device = m, device
        return self._model

    def rerank(
        self, query: str, papers: Sequence[Paper]
    ) -> tuple[list[Paper], RerankStats]:
        """`papers` 를 쿼리 기준으로 다시 세웁니다. 상위 `top_n` 만 대상입니다."""
        cfg = self.config
        stats = RerankStats(n_in=len(papers), model=cfg.model)
        t0 = time.perf_counter()

        if not papers:
            stats.note = "후보 없음"
            return [], stats

        head = list(papers[: cfg.top_n])
        tail = list(papers[cfg.top_n :])
        stats.n_untouched = len(tail)

        try:
            model = self.model
            pairs = [(query, _doc_text(p, cfg.abstract_chars)) for p in head]
            scores = model.predict(pairs, batch_size=cfg.batch_size,
                                   show_progress_bar=False)
        except Exception as e:  # noqa: BLE001 — 재랭킹 실패로 검색 전체를 죽이지 않습니다
            stats.errors.append(f"{type(e).__name__}: {e}")
            stats.note = "재랭킹 실패 -> 원래 순서 유지"
            stats.elapsed_s = time.perf_counter() - t0
            log.warning("cross-encoder 재랭킹 실패, 원래 순서 유지: %s", e)
            return list(papers), stats

        stats.applied = True
        stats.n_scored = len(head)
        stats.device = self._device or ""

        # cross 점수는 스케일이 제각각(로짓)이라 원래 점수와 직접 못 섞습니다.
        # 순위 기반으로 정규화한 뒤 섞습니다 — 파이프라인의 다른 곳과 같은 이유입니다.
        order = sorted(range(len(head)), key=lambda i: -float(scores[i]))
        cross_rank = {i: r for r, i in enumerate(order)}
        n = max(len(head) - 1, 1)

        blended = []
        for i, p in enumerate(head):
            cross = 1.0 - cross_rank[i] / n          # 1.0 = cross 가 1위
            orig = 1.0 - i / n                        # 1.0 = 원래 1위
            blended.append((p, (1 - cfg.blend) * cross + cfg.blend * orig))
        blended.sort(key=lambda kv: (-kv[1], kv[0].paper_id))

        out = [
            Paper(**{**p.__dict__, "score": s,
                     "provenance": tuple(dict.fromkeys((*p.provenance, "rerank")))})
            for p, s in blended
        ]
        out.extend(tail)

        stats.elapsed_s = time.perf_counter() - t0
        stats.note = (f"top_n={cfg.top_n} blend={cfg.blend} "
                      f"scored={stats.n_scored} untouched={stats.n_untouched}")
        return out, stats
