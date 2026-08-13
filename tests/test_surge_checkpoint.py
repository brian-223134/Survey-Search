"""SurGE 평가의 체크포인트 — 두 시간짜리 실행을 안 날리기 위한 안전장치.

백엔드도 검색도 안 씁니다. `run()` 이 이미 끝난 (설정,토픽) 쌍을 다시 계산하지 않는가,
중간에 죽어도 남는가, 조건이 다른 결과를 조용히 섞지 않는가만 봅니다.
"""

from __future__ import annotations

import json

import pytest

from survey_search.eval.surge import GoldTopic, _ckpt_fingerprint, run

CONFIGS = {"a": dict(lexical=False), "b": dict()}


def gold(sid: int) -> GoldTopic:
    return GoldTopic(survey_id=sid, topic=f"topic {sid}", year=2021, date="2021-06-01",
                     gold_ids=[f"g{sid}"], n_cites_total=1, n_matched=1)


@pytest.fixture
def spy(monkeypatch):
    """search_topic 을 가로채 호출 횟수를 셉니다. 정답을 그대로 돌려줘 recall=1 이 됩니다."""
    import survey_search.search as S

    calls: list[tuple[str, str]] = []

    class R:
        def __init__(self, ids): self._ids = ids
        def ids(self): return self._ids

    def fake(topic, *, backend, config):
        calls.append(topic)
        return R([f"g{topic.split()[-1]}"])

    monkeypatch.setattr(S, "search_topic", fake)
    return calls


def test_resume_skips_finished_pairs_and_keeps_scores(tmp_path, spy):
    ck = tmp_path / "ck.jsonl"
    topics = [gold(1), gold(2)]

    first = run(topics, backend=None, configs=CONFIGS, checkpoint=ck)
    assert len(spy) == 4          # 토픽 2 × 설정 2
    spy.clear()

    second = run(topics, backend=None, configs=CONFIGS, checkpoint=ck)
    assert spy == []              # 전부 이어받아 재계산 0회
    for name in CONFIGS:
        assert [s.survey_id for s in second[name]] == [s.survey_id for s in first[name]]
        assert [s.ndcg for s in second[name]] == [s.ndcg for s in first[name]]


def test_partial_run_resumes_from_where_it_died(tmp_path, spy, monkeypatch):
    """3번째 쌍에서 죽여도 앞의 2개는 살아 있어야 합니다."""
    import survey_search.search as S

    ck = tmp_path / "ck.jsonl"
    topics = [gold(1), gold(2)]
    real = S.search_topic

    def die_on_third(topic, *, backend, config):
        if len(spy) >= 2:
            raise RuntimeError("죽는 시늉")
        return real(topic, backend=backend, config=config)

    monkeypatch.setattr(S, "search_topic", die_on_third)
    with pytest.raises(RuntimeError):
        run(topics, backend=None, configs=CONFIGS, checkpoint=ck)

    assert len(ck.read_text().splitlines()) == 1 + 2     # 지문 1줄 + 끝난 쌍 2줄

    monkeypatch.setattr(S, "search_topic", real)
    spy.clear()
    out = run(topics, backend=None, configs=CONFIGS, checkpoint=ck)
    assert len(spy) == 2                                  # 남은 2쌍만 계산
    assert all(len(v) == 2 for v in out.values())


def test_different_experiment_conditions_refuse_to_merge(tmp_path, spy):
    """설정이 다른 결과가 한 표에 섞이면 표가 조용히 거짓말을 합니다."""
    ck = tmp_path / "ck.jsonl"
    run([gold(1)], backend=None, configs=CONFIGS, checkpoint=ck)

    with pytest.raises(ValueError, match="다른 실험 조건"):
        run([gold(1)], backend=None, configs={"a": dict(freshness=True), "b": dict()},
            checkpoint=ck)
    with pytest.raises(ValueError, match="다른 실험 조건"):
        run([gold(1)], backend=None, configs=CONFIGS, checkpoint=ck, n_papers=500)


def test_truncated_last_line_is_dropped_not_fatal(tmp_path, spy):
    """kill -9 는 줄 중간에 떨어질 수 있습니다. 그 쌍만 다시 계산하면 됩니다."""
    ck = tmp_path / "ck.jsonl"
    run([gold(1)], backend=None, configs=CONFIGS, checkpoint=ck)
    with ck.open("a") as fh:
        fh.write('{"config": "a", "survey_id": 2, "top')     # 잘린 줄
    spy.clear()

    out = run([gold(1)], backend=None, configs=CONFIGS, checkpoint=ck)
    assert spy == []
    assert all(len(v) == 1 for v in out.values())


def test_recall_cutoff_keys_survive_the_json_round_trip(tmp_path, spy):
    """recall 키는 int 인데 JSON 은 문자열로만 씁니다. 안 되돌리면 집계가 KeyError 로 죽습니다."""
    from survey_search.eval.surge import aggregate

    ck = tmp_path / "ck.jsonl"
    run([gold(1)], backend=None, configs=CONFIGS, checkpoint=ck)
    out = run([gold(1)], backend=None, configs=CONFIGS, checkpoint=ck)
    assert all(isinstance(k, int) for k in out["a"][0].recall)
    assert aggregate(out["a"])["recall"]


def test_no_checkpoint_writes_nothing(tmp_path, spy):
    """기본 동작은 그대로 — 경로를 안 주면 파일을 안 만듭니다."""
    run([gold(1)], backend=None, configs=CONFIGS)
    assert list(tmp_path.iterdir()) == []


def test_fingerprint_is_stable_across_key_order():
    a = _ckpt_fingerprint({"x": {"p": 1, "q": 2}, "y": {}}, 1500, True)
    b = _ckpt_fingerprint({"y": {}, "x": {"q": 2, "p": 1}}, 1500, True)
    assert a == b


def test_header_is_written_before_any_result(tmp_path, spy):
    ck = tmp_path / "ck.jsonl"
    run([gold(1)], backend=None, configs=CONFIGS, checkpoint=ck)
    assert "_fingerprint" in json.loads(ck.read_text().splitlines()[0])
