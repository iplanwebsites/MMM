"""Tests for the inference of an MMM model."""

from pathlib import Path

import pytest
from datasets import load_dataset
from miditok import MMM
from symusic import Score

from mmm import DatasetMMM
from utils.constants import (
    ACS_RANDOM_RATIO_RANGE,
    BARS_IDX_RANDOM_RATIO_RANGE,
    DATA_AUGMENTATION_OFFSETS,
    MAX_SEQ_LEN,
    RATIO_BAR_INFILLING,
    RATIOS_RANGE_BAR_INFILLING_DURATION,
    TRACKS_IDX_RANDOM_RATIO_RANGE,
    TRACKS_SELECTION_RANDOM_RATIO_RANGE,
)

from .utils_tests import MIDI_PATHS


@pytest.mark.parametrize(
    "tokenizer",
    [MMM(params=Path("runs", "tokenizer.json"))],
)
def test_dataset(tokenizer: MMM):
    dataset = load_dataset()  # TODO create from MIDI files
    dataset_obj = DatasetMMM(
        dataset,
        tokenizer,
        MAX_SEQ_LEN,
        TRACKS_SELECTION_RANDOM_RATIO_RANGE,
        DATA_AUGMENTATION_OFFSETS,
        RATIO_BAR_INFILLING,
        RATIOS_RANGE_BAR_INFILLING_DURATION,
        ac_random_ratio_range=ACS_RANDOM_RATIO_RANGE,
        ac_tracks_random_ratio_range=TRACKS_IDX_RANDOM_RATIO_RANGE,
        ac_bars_random_ratio_range=BARS_IDX_RANDOM_RATIO_RANGE,
    )

    for filepath in MIDI_PATHS:
        score = Score(filepath)
        # score = dataset_obj.augment_and_preprocess_score(score)
        tokseq, decoder_input = dataset_obj._tokenize_score(score)
        print(tokseq)
