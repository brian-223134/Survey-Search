"""S8 스노우볼링 — mock 백엔드, 네트워크 불필요."""

import pytest

from survey_search.core.expand import SnowballConfig, snowball
from survey_search.types import Paper


def mk(pid: str, score: float = 1.0) -> Paper:
    return Paper(paper_id=pid, base_id=pid, title=pid, abstract="",
                 date="2025-01-01", score=score)


class MockCitationBackend:
    """references/cited_by 를 아는 백엔드."""

    name = "mock"

    def __init__(self, refs: dict, cits: dict, known: set[str] | None = None) -> None:
        self.refs, self.cits = refs, cits
        self.known = known
        self.calls = 0

    def references(self, pid):
        self.calls += 1
        return self.refs.get(pid, [])

    def cited_by(self, pid):
        self.calls += 1
        return self.cits.get(pid, [])

    def get_papers(self, ids):
        pool = self.known if self.known is not None else set(ids)
        return [mk(i) for i in ids if i in pool]


class MockPlainBackend:
    """인용 엣지를 모르는 백엔드 — 로컬 FAISS 백엔드가 이 경우입니다."""

    name = "plain"

    def get_papers(self, ids):
        return [mk(i) for i in ids]


def test_noop_when_backend_has_no_citation_edges():
    """조용히 통과시키지 말고 supported=False 로 남겨야 합니다."""
    out, stats = snowball([mk("a")], backend=MockPlainBackend())
    assert out == []
    assert stats.supported is False
    assert "인용 엣지" in stats.note


def test_pulls_in_neighbours_not_already_in_candidates():
    be = MockCitationBackend(refs={"a": ["old1", "old2"]}, cits={"a": ["new1"]})
    out, stats = snowball([mk("a")], backend=be)
    ids = {p.paper_id for p in out}
    assert ids == {"old1", "old2", "new1"}
    assert stats.n_backward == 2 and stats.n_forward == 1
    assert stats.n_new == 3


def test_does_not_re_add_existing_candidates():
    be = MockCitationBackend(refs={"a": ["b"]}, cits={})
    out, _ = snowball([mk("a"), mk("b")], backend=be)
    assert out == []


def test_min_seed_support_filters_singletons():
    """두 시드가 공통으로 가리킨 논문만 남기는 모드."""
    be = MockCitationBackend(refs={"a": ["shared", "only_a"], "b": ["shared"]}, cits={})
    out, stats = snowball([mk("a"), mk("b")], backend=be,
                          config=SnowballConfig(min_seed_support=2))
    assert {p.paper_id for p in out} == {"shared"}
    assert stats.n_dropped_by_support == 1


def test_support_ranks_output():
    be = MockCitationBackend(refs={"a": ["x", "y"], "b": ["x"]}, cits={})
    out, _ = snowball([mk("a"), mk("b")], backend=be)
    assert out[0].paper_id == "x"      # 두 시드가 가리킴


def test_max_new_caps_and_counts_the_drop():
    be = MockCitationBackend(refs={"a": [f"r{i}" for i in range(50)]}, cits={})
    out, stats = snowball([mk("a")], backend=be, config=SnowballConfig(max_new=10))
    assert len(out) == 10
    assert stats.n_dropped_by_cap == 40


def test_n_seeds_limits_how_many_papers_are_expanded():
    be = MockCitationBackend(refs={f"p{i}": [f"r{i}"] for i in range(10)}, cits={})
    cands = [mk(f"p{i}") for i in range(10)]
    out, stats = snowball(cands, backend=be, config=SnowballConfig(n_seeds=3))
    assert stats.n_seeds == 3
    assert {p.paper_id for p in out} == {"r0", "r1", "r2"}


def test_backward_and_forward_can_be_disabled():
    be = MockCitationBackend(refs={"a": ["back"]}, cits={"a": ["fwd"]})
    out, _ = snowball([mk("a")], backend=be, config=SnowballConfig(forward=False))
    assert {p.paper_id for p in out} == {"back"}
    out, _ = snowball([mk("a")], backend=be, config=SnowballConfig(backward=False))
    assert {p.paper_id for p in out} == {"fwd"}


def test_missing_metadata_is_counted_not_hidden():
    """엣지에는 나왔는데 메타데이터가 없는 논문 — 조용히 사라지면 안 됩니다."""
    be = MockCitationBackend(refs={"a": ["has_meta", "no_meta"]}, cits={},
                             known={"has_meta"})
    out, stats = snowball([mk("a")], backend=be)
    assert {p.paper_id for p in out} == {"has_meta"}
    assert stats.n_meta_missing == 1


def test_backend_error_on_one_seed_does_not_kill_the_stage():
    class Flaky(MockCitationBackend):
        def references(self, pid):
            if pid == "bad":
                raise RuntimeError("boom")
            return super().references(pid)

    be = Flaky(refs={"good": ["r1"]}, cits={})
    out, stats = snowball([mk("bad"), mk("good")], backend=be)
    assert {p.paper_id for p in out} == {"r1"}
    assert stats.errors and "boom" in stats.errors[0]


def test_provenance_is_marked_snowball():
    be = MockCitationBackend(refs={"a": ["r"]}, cits={})
    out, _ = snowball([mk("a")], backend=be)
    assert out[0].provenance == ("snowball",)


def test_hops_zero_does_nothing():
    be = MockCitationBackend(refs={"a": ["r"]}, cits={})
    out, stats = snowball([mk("a")], backend=be, config=SnowballConfig(hops=0))
    assert out == []
    assert stats.supported is True
