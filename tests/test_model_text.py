"""What the model reads, in training and in the service, is the same text."""

from __future__ import annotations

import pytest

from fintext import preprocess
from fintext.preprocess import fold_for_model, model_text


class TestFold:
    def test_alef_variants_become_bare_alef(self):
        assert fold_for_model("أحمد إلى آخر ٱلكتاب") == "احمد الى اخر الكتاب"

    def test_elongation_keeps_two(self):
        assert fold_for_model("هههههه soooo") == "هه soo"

    def test_numbers_are_never_shortened(self):
        text = "Revenue rose to 1000000 from 2000, up 333%"
        assert fold_for_model(text) == text

    @pytest.mark.parametrize("text", ["أأأأإإ", "hellooo  ههههه", "plain text", ""])
    def test_idempotent(self, text):
        once = fold_for_model(text)
        assert fold_for_model(once) == once


class TestTrainingMatchesServing:
    @pytest.mark.parametrize(
        "raw",
        [
            "يا  غبيييي​ أنت ",  # zero-width space, elongation, hamza alef
            "Call me on +971 50 123 4567, sooo good",
            "مُحَمَّد",  # diacritics
            "@USER URL كل زق",
        ],
    )
    def test_model_text_is_what_the_service_feeds_the_model(self, raw):
        served = fold_for_model(preprocess.prepare(raw).text)
        assert model_text(raw) == served

    def test_pii_is_scrubbed_in_training_too(self):
        assert "123 4567" not in model_text("call +971 50 123 4567 now")


@pytest.mark.parametrize("on_model_text", [True, False])
def test_collator_trains_on_raw_text_only_when_asked(on_model_text):
    pytest.importorskip("torch")
    from fintext.preprocess import model_text
    from fintext.training import _collator

    seen = []

    class Tokenizer:
        def __call__(self, texts, **_):
            seen.extend(texts)
            raise StopIteration  # the texts are all this test needs

    raw = "غبيييي يا أحمد"
    collate = _collator(Tokenizer(), 32, "cpu", on_model_text=on_model_text)
    with pytest.raises(StopIteration):
        collate([{"text": raw, "language": "ar", "labels": {}}])
    assert seen == [model_text(raw) if on_model_text else raw]
    assert model_text(raw) != raw
