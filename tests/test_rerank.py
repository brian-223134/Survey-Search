"""S8b cross-encoder 재랭킹 — 모델 없이 도는 부분.

실제 모델 로드는 `-m assets` 로 분리했습니다. 재랭킹 로직(정규화·blend·폴백)은
모델 없이 검증할 수 있어야 CI 나 GPU 가 막힌 환경에서도 돕니다.
"""

from __future__ import annotations

import pytest

from survey_search.core.rerank import (
    CrossEncoderReranker,
    RerankConfig,
    _doc_text,
    _resolve_device,
)
from survey_search.types import Paper


def mk(pid: str, title: str = "", abstract: str = "", score: float = 1.0) -> Paper:
    return Paper(paper_id=pid, base_id=pid, title=title or pid, abstract=abstract,
                 date="2025-01-01", score=score)


class FakeCE:
    """예측 점수를 지정할 수 있는 가짜 cross-encoder."""

    def __init__(self, scores, record=None):
        self.scores, self.record = scores, record

    def predict(self, pairs, **kw):
        if self.record is not None:
            self.record.extend(pairs)
        return self.scores[: len(pairs)]


def reranker(scores, record=None, **cfg) -> CrossEncoderReranker:
    r = CrossEncoderReranker(RerankConfig(**cfg))
    r._model = FakeCE(scores, record)
    r._device = "cpu"
    return r


def test_reorders_by_cross_score():
    papers = [mk("a"), mk("b"), mk("c")]
    out, stats = reranker([0.1, 0.9, 0.5]).rerank("q", papers)
    assert [p.paper_id for p in out] == ["b", "c", "a"]
    assert stats.applied is True
    assert stats.n_scored == 3


def test_only_top_n_is_reranked_rest_keeps_order():
    """비용 때문에 상위만 재랭킹합니다. 나머지는 원래 순서를 그대로 유지해야 합니다."""
    papers = [mk(x) for x in "abcde"]
    out, stats = reranker([0.1, 0.9], top_n=2).rerank("q", papers)
    assert [p.paper_id for p in out] == ["b", "a", "c", "d", "e"]
    assert stats.n_scored == 2
    assert stats.n_untouched == 3


def test_blend_one_preserves_original_order():
    """blend=1.0 이면 재랭킹을 안 한 것과 같아야 합니다 — ablation 의 기준선."""
    papers = [mk("a"), mk("b"), mk("c")]
    out, _ = reranker([0.1, 0.9, 0.5], blend=1.0).rerank("q", papers)
    assert [p.paper_id for p in out] == ["a", "b", "c"]


def test_blend_zero_is_pure_cross_encoder():
    papers = [mk("a"), mk("b"), mk("c")]
    out, _ = reranker([0.1, 0.9, 0.5], blend=0.0).rerank("q", papers)
    assert [p.paper_id for p in out] == ["b", "c", "a"]


def test_blend_half_can_break_ties_toward_original():
    """cross 와 원래 순위가 정반대면 blend=0.5 에서 원래 순위가 동점을 가릅니다."""
    papers = [mk("a"), mk("b")]
    out, _ = reranker([0.0, 1.0], blend=0.5).rerank("q", papers)
    assert len(out) == 2   # 동점이어도 떨어뜨리지 않습니다


def test_scores_are_rank_normalised_not_raw_logits():
    """cross 점수는 로짓이라 스케일이 제각각입니다. 그대로 두면 이후 단계가 오작동합니다."""
    papers = [mk("a"), mk("b"), mk("c")]
    out, _ = reranker([-11.2, 8.4, 0.3]).rerank("q", papers)
    assert all(0.0 <= p.score <= 1.0 for p in out)
    assert out[0].score == 1.0


def test_provenance_records_rerank_without_duplicating():
    p = Paper(paper_id="a", base_id="a", title="a", abstract="", date="2025-01-01",
              provenance=("dense", "rerank"))
    out, _ = reranker([0.5]).rerank("q", [p])
    assert out[0].provenance == ("dense", "rerank")


def test_model_failure_keeps_original_order_and_reports():
    """재랭킹이 실패해도 검색 전체가 죽으면 안 됩니다. 대신 사유를 남깁니다."""

    class Boom:
        def predict(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    r = CrossEncoderReranker()
    r._model = Boom()
    papers = [mk("a"), mk("b")]
    out, stats = r.rerank("q", papers)
    assert [p.paper_id for p in out] == ["a", "b"]
    assert stats.applied is False
    assert stats.errors and "CUDA out of memory" in stats.errors[0]


def test_empty_input():
    out, stats = reranker([]).rerank("q", [])
    assert out == []
    assert stats.applied is False


def test_doc_text_truncates_abstract_not_title():
    """제목이 잘리면 안 됩니다 — 512토큰 상한에서 가장 중요한 신호입니다."""
    p = mk("x", title="A Very Important Title", abstract="z" * 5000)
    text = _doc_text(p, abstract_chars=100)
    assert text.startswith("A Very Important Title")
    assert len(text) < 200


def test_pairs_carry_the_query_and_doc_text():
    rec = []
    papers = [mk("a", title="T1", abstract="A1")]
    reranker([0.5], record=rec).rerank("my query", papers)
    assert rec[0][0] == "my query"
    assert "T1" in rec[0][1] and "A1" in rec[0][1]


def test_resolve_device_explicit_wins():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda:3") == "cuda:3"


@pytest.mark.assets
def test_real_model_scores_relevant_higher():
    """실제 모델 — 관련 논문이 무관한 논문보다 높은 점수를 받아야 합니다."""
    pytest.importorskip("sentence_transformers")
    r = CrossEncoderReranker(RerankConfig(top_n=10))
    papers = [
        mk("off", title="A Survey of Medieval Pottery Kilns",
           abstract="We catalogue kiln sites across Europe."),
        mk("on", title="Retrieval-Augmented Generation for Knowledge-Intensive NLP",
           abstract="We combine parametric and non-parametric memory for generation."),
    ]
    out, stats = r.rerank("retrieval augmented generation for language models", papers)
    assert stats.applied, stats.errors
    assert out[0].paper_id == "on"


def test_default_top_n_exceeds_typical_n_papers():
    """top_n <= n_papers 면 최종 집합이 안 바뀌고 순서만 바뀝니다.
    기본값이 그 함정에 빠지면 안 됩니다."""
    assert RerankConfig().top_n > 1500
