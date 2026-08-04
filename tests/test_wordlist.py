"""Structured word-list resolution + matching (profanity-list style lookups).

_load_word_list is lru_cached, so every test uses a distinct file path/ref to
avoid cross-test cache bleed. No DB: the missing-ref branch is exercised with a
stubbed SessionLocal so nothing touches Postgres.
"""
import json

import pytest

from tools import retrieval
from tools.retrieval import _load_word_list, lookup_structured


# --- _load_word_list: file formats -----------------------------------------

def test_load_json_array(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text(json.dumps(["alpha", "beta"]), encoding="utf-8")
    assert _load_word_list(str(p)) == ("alpha", "beta")


def test_load_json_words_object(tmp_path):
    p = tmp_path / "obj.json"
    p.write_text(json.dumps({"words": ["one", "two"]}), encoding="utf-8")
    assert _load_word_list(str(p)) == ("one", "two")


def test_load_newline_delimited(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("foo\n  bar  \n\nbaz\n", encoding="utf-8")
    assert _load_word_list(str(p)) == ("foo", "bar", "baz")


def test_missing_ref_resolves_to_empty(monkeypatch):
    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, model, ref):
            return None

    monkeypatch.setattr(retrieval, "SessionLocal", lambda: _FakeSession())
    assert _load_word_list("/no/such/wordlist-xyz.json") == ()


# --- lookup_structured: matching semantics ----------------------------------

def _wordlist(tmp_path, name, words):
    p = tmp_path / name
    p.write_text(json.dumps(words), encoding="utf-8")
    return str(p)


def test_match_is_case_insensitive(tmp_path):
    ref = _wordlist(tmp_path, "ci.json", ["Jackpot"])
    assert lookup_structured(ref, "won the JACKPOT today") is True
    assert lookup_structured(ref, "no gambling here") is False


def test_word_boundary_blocks_substring_false_positive(tmp_path):
    ref = _wordlist(tmp_path, "wb.json", ["ass"])
    # "class" must not count as a hit for the term "ass"
    assert lookup_structured(ref, "what a great class") is False
    assert lookup_structured(ref, "you ass") is True


def test_punctuation_counts_as_boundary(tmp_path):
    ref = _wordlist(tmp_path, "punc.json", ["bet"])
    assert lookup_structured(ref, "place a bet, now!") is True


def test_non_ascii_term_matches(tmp_path):
    ref = _wordlist(tmp_path, "unicode.json", ["바보", "café"])
    assert lookup_structured(ref, "너 정말 바보 같아") is True
    assert lookup_structured(ref, "met at the café earlier") is True
    assert lookup_structured(ref, "완전 천재") is False


@pytest.mark.parametrize("ref,text", [("", "some text"), ("ref", ""), ("", "")])
def test_empty_ref_or_text_short_circuits_false(ref, text):
    # early return before any file/DB access
    assert lookup_structured(ref, text) is False


def test_empty_wordlist_file_yields_no_match(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("[]", encoding="utf-8")
    assert lookup_structured(str(p), "anything at all") is False
