"""Card copy stays short and drops retriever jargon. The stored sentence is untouched."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPACE = Path(__file__).resolve().parents[1] / "space"


def _phrases():
    path = SPACE / "phrases.py"
    spec = importlib.util.spec_from_file_location("space_phrases_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plain_reasons_keep_two_specific_lines_and_drop_the_jargon():
    phrases = _phrases()
    reasons = [
        "Ranked highly by the neural retriever for your taste profile",
        "Popular with many viewers",
        "Because you rated Toy Story highly",
        "You rated Toy Story 5★ and this is one of its nearest neighbours",
        "Lines up with movies you have already rated",
    ]
    shown = phrases.plain_reasons(reasons)
    assert len(shown) == 2
    assert shown[0] == "Matches films you tend to like"
    assert shown[1] == "Because you rated Toy Story 5★"
    assert all("neural" not in line and "neighbour" not in line for line in shown)


def test_a_generic_line_fills_the_card_when_nothing_specific_was_recorded():
    phrases = _phrases()
    shown = phrases.plain_reasons(["Popular with many viewers", "Widely rated by other viewers"])
    assert shown == ["Popular with many viewers", "Lots of people have rated this"]


def test_method_sentence_does_not_name_the_retriever():
    phrases = _phrases()
    text = phrases.plain_method("cold_start_ranker")
    assert "neural" not in text
    assert "N=" not in text
