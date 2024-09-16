"""Tests for the inference of an MMM model."""

from pathlib import Path

import pytest
from datasets import Dataset
from miditok import MMM

from mmm import DatasetMMM

from .utils_tests import MIDI_PATHS

MAX_SEQ_LEN = 2048


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
        MAX_SEQ_LEN,
        (0.4, 1),
        (6, 2, 0),
        0.75,
        (0.1, 0.4),
        ac_random_ratio_range=(0.05, 0.9),
        ac_tracks_random_ratio_range=(0.1, 1),
        ac_bars_random_ratio_range=(0.1, 0.7),
    )
    for sample in dataset_obj:
        if (
            sample["input_ids"] is not None
            and (num_tokens := len(sample["input_ids"])) > MAX_SEQ_LEN
        ):
            msg = (
                f"The returned token ids sequence must contain fewer than "
                f"{MAX_SEQ_LEN} tokens (found {num_tokens}"
            )
            raise ValueError(msg)
