"""S1 facet 분해 — 네트워크 없이 도는 부분만."""

import json
import time
import urllib.error

import pytest

from survey_search.core.facets import (
    FacetConfig,
    _parse_facets,
    cache_key,
    decompose,
    fallback_facets,
    load_dotenv,
)


def cfg(**kw) -> FacetConfig:
    return FacetConfig(**kw).resolved()


def test_parse_plain_json():
    out = _parse_facets('{"facets":[{"name":"A","queries":["q1","q2"]}]}', cfg())
    assert out[0].name == "A"
    assert out[0].queries == ("q1", "q2")


def test_parse_survives_markdown_fence():
    text = '```json\n{"facets":[{"name":"A","queries":["q"]}]}\n```'
    assert _parse_facets(text, cfg())[0].name == "A"


def test_parse_survives_surrounding_prose():
    text = 'Sure! Here you go:\n{"facets":[{"name":"A","queries":["q"]}]}\nHope this helps.'
    assert _parse_facets(text, cfg())[0].name == "A"


def test_parse_drops_facets_without_queries():
    out = _parse_facets(
        '{"facets":[{"name":"A","queries":[]},{"name":"B","queries":["q"]}]}', cfg()
    )
    assert [f.name for f in out] == ["B"]


def test_parse_caps_queries_per_facet():
    c = FacetConfig(max_queries_per_facet=2).resolved()
    out = _parse_facets('{"facets":[{"name":"A","queries":["1","2","3","4"]}]}', c)
    assert len(out[0].queries) == 2


def test_parse_raises_on_garbage():
    with pytest.raises(ValueError):
        _parse_facets("no json at all", cfg())


def test_parse_raises_when_no_facets_survive():
    with pytest.raises(ValueError):
        _parse_facets('{"facets":[]}', cfg())


def test_fallback_produces_query_variants():
    out = fallback_facets("Retrieval-Augmented Generation for Large Language Models")
    assert len(out) == 1
    assert out[0].name == "(fallback)"
    qs = out[0].queries
    assert len(qs) >= 2
    assert qs[0] == "Retrieval-Augmented Generation for Large Language Models"
    # 불용어를 뺀 축약형이 들어가야 합니다
    assert any("for" not in q for q in qs[1:])


def test_fallback_never_empty_even_for_one_word():
    assert fallback_facets("transformers")[0].queries


def test_cache_key_is_stable_and_case_insensitive():
    c = cfg()
    assert cache_key("RAG for LLMs", c) == cache_key("  rag for llms  ", c)


def test_cache_key_changes_with_model():
    a = FacetConfig(model="m1").resolved()
    b = FacetConfig(model="m2").resolved()
    assert cache_key("t", a) != cache_key("t", b)


def test_decompose_falls_back_without_key_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    c = FacetConfig(api_key="", cache_dir=tmp_path)
    facets, stats = decompose("Some Topic", config=c)
    assert stats.source == "fallback"
    assert stats.llm_calls == 0
    assert stats.warnings, "fallback 으로 내려간 사실이 stats 에 남아야 합니다"
    assert facets


def test_fallback_is_not_cached(tmp_path, monkeypatch):
    """키를 넣고 다시 돌리면 제대로 나와야 하므로 fallback 은 캐시하면 안 됩니다."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    c = FacetConfig(api_key="", cache_dir=tmp_path)
    decompose("Some Topic", config=c)
    assert list(tmp_path.glob("*.json")) == []


def test_cache_hit_avoids_llm(tmp_path):
    c = FacetConfig(api_key="dummy", cache_dir=tmp_path).resolved()
    key = cache_key("Cached Topic", c)
    (tmp_path / f"{key}.json").write_text(json.dumps(
        {"topic": "Cached Topic", "facets": [{"name": "F", "queries": ["q"]}]}
    ))
    facets, stats = decompose("Cached Topic", config=c)
    assert stats.source == "cache"
    assert stats.llm_calls == 0
    assert facets[0].name == "F"


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("export FOO=fromfile\nBAR=alsofile\n# comment\n\n")
    monkeypatch.setenv("FOO", "fromenv")
    load_dotenv(p)
    import os

    assert os.environ["FOO"] == "fromenv"   # 기존 값이 이깁니다
    assert os.environ["BAR"] == "alsofile"


def test_load_dotenv_missing_file_is_noop():
    assert load_dotenv("/nonexistent/.env") == 0


def test_parse_rejects_empty_content_instead_of_crashing():
    """OpenRouter 가 content=None 을 돌려주는 일이 실제로 있습니다.
    AttributeError 로 터지면 decompose 의 fallback 이 못 잡아 배치 전체가 죽습니다."""
    for bad in (None, "", "   ", "\n"):
        with pytest.raises(ValueError, match="빈 응답"):
            _parse_facets(bad, cfg())


def test_decompose_falls_back_on_any_llm_error(tmp_path, monkeypatch):
    """어떤 예외든 fallback 으로 흡수하고 사유를 남겨야 합니다."""
    import survey_search.core.facets as F

    def boom(topic, cfg):
        raise AttributeError("'NoneType' object has no attribute 'strip'")

    monkeypatch.setattr(F, "_call_openrouter", boom)
    facets, stats = F.decompose("T", config=FacetConfig(api_key="k", cache_dir=tmp_path))
    assert stats.source == "fallback"
    assert facets
    assert any("AttributeError" in w for w in stats.warnings)


# --- 벽시계 상한 (keep-alive 로 소켓 타임아웃이 안 걸리는 실제 사고에 대한 대응) ------

class _KeepAliveStream:
    """응답을 안 끝내면서 주기적으로 바이트만 흘리는 서버 흉내.

    실측한 실제 동작입니다 — OpenRouter 가 10초마다 912바이트를 보내는 바람에
    `urlopen(timeout=90)` 이 매 읽기마다 초기화돼 16분 넘게 안 끝났습니다.
    """

    def read(self):
        while True:
            time.sleep(0.01)      # 읽기 자체는 계속 성공 = 소켓 타임아웃 발동 안 함

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_deadline_fires_even_when_socket_timeout_cannot(monkeypatch):
    """소켓 타임아웃이 구조적으로 못 잡는 정지를 벽시계 상한이 잡아야 합니다."""
    import survey_search.core.facets as F

    monkeypatch.setattr(F.urllib.request, "urlopen",
                        lambda *a, **k: _KeepAliveStream())
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError, match="안 끝났습니다"):
        F._read_with_deadline(object(), cfg(deadline_s=0.3, timeout_s=90.0))
    assert time.perf_counter() - t0 < 5.0   # 90초 상한을 안 기다려야 합니다


def test_deadline_propagates_the_real_error_not_a_timeout(monkeypatch):
    """스레드 안에서 난 예외는 원형 그대로 올라와야 재시도 분기가 맞습니다."""
    import survey_search.core.facets as F

    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(F.urllib.request, "urlopen", boom)
    with pytest.raises(urllib.error.URLError):
        F._read_with_deadline(object(), cfg())


def test_call_openrouter_retries_then_gives_up_on_deadline(monkeypatch):
    """상한 초과는 재시도 대상이고, 다 실패하면 RuntimeError -> fallback 으로 갑니다."""
    import survey_search.core.facets as F

    calls = []

    def hang(req, c):
        calls.append(1)
        raise TimeoutError("응답이 0초 안에 안 끝났습니다")

    monkeypatch.setattr(F, "_read_with_deadline", hang)
    monkeypatch.setattr(F.time, "sleep", lambda s: None)   # 백오프 대기 건너뜁니다
    with pytest.raises(RuntimeError, match="3회 모두 실패"):
        F._call_openrouter("T", cfg(api_key="k"))
    assert len(calls) == 3


def test_deadline_default_is_bounded():
    """기본값이 None/무한이면 이 사고가 그대로 재발합니다."""
    d = FacetConfig().resolved().deadline_s
    assert 0 < d <= 600
