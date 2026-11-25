"""A training seed reseeds training without touching the split."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from fintext import training  # noqa: E402
from fintext.intervals import load_scored  # noqa: E402


class _Stop(Exception):
    pass


@pytest.mark.parametrize("training_seed,expected", [(None, 20240101), (7, 7)])
def test_train_seeds_from_training_seed_when_set(monkeypatch, training_seed, expected):
    seen = []

    def record(seed):
        seen.append(seed)
        raise _Stop

    monkeypatch.setattr(training, "set_seed", record)
    config = training.TrainingConfig(training_seed=training_seed)
    with pytest.raises(_Stop):
        training.train(config, {"train": [], "dev": [], "test": []})
    assert seen == [expected]
    assert config.seed == 20240101  # the split seed is untouched


def test_scored_file_round_trips(tmp_path):
    records = [{"key": "a", "language": "en", "toxicity_true": 1, "toxicity_score": 0.9}]
    where = tmp_path / "scored_test.jsonl"
    training.write_scored(where, "run-x", {"toxicity_en": 0.5}, records)
    header, rows = load_scored(where)
    assert header == {"thresholds": {"toxicity_en": 0.5}, "run": "run-x"}
    assert rows == records
