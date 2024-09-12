"""Tests for the inference of an MMM model."""

from pathlib import Path

import pytest
from datasets import Dataset
from miditok import MMM
from symusic import Score

from mmm import DatasetMMM

from .utils_tests import MIDI_PATHS


def gen():
    for file_path in MIDI_PATHS:
        yield {"music": {"bytes": file_path.read_bytes()}}


@pytest.mark.parametrize(
    "tokenizer",
    [MMM(params=Path("runs", "tokenizer.json"))],
)
def test_dataset(tokenizer: MMM):
    dataset = Dataset.from_generator(gen)
    dataset_obj = DatasetMMM(
        dataset,
        tokenizer,
        2048,
        (0.4, 1),
        (6, 2, 0),
        0.75,
        (0.1, 0.4),
        ac_random_ratio_range=(0.05, 0.9),
        ac_tracks_random_ratio_range=(0.1, 1),
        ac_bars_random_ratio_range=(0.1, 0.7),
    )

    for filepath in MIDI_PATHS:
        score = Score(filepath)
        # score = dataset_obj.augment_and_preprocess_score(score)
        tokseq, decoder_input = dataset_obj._tokenize_score(score)
        print(tokseq)
