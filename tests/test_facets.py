"""S1 facet 분해 — 네트워크 없이 도는 부분만."""

import json

import pytest

from survey_search.core.facets import (
    FacetConfig,
    _parse_facets,
    cache_key,
    decompose,
    fallback_facets,
    load_dotenv,
)


def cfg() -> FacetConfig:
    return FacetConfig().resolved()


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
