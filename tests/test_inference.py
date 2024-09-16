"""Tests for MMM inference."""

from __future__ import annotations

from pathlib import Path

import pytest
from miditok import MMM

from mmm import InferenceConfig, generate

from .utils_tests import MIDI_PATHS, DummyModel

INFERENCE_CONFIG = InferenceConfig(
    {
        0: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
        # 2: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
        # 3: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
    },
    [
        (43, ["ACTrackOnsetPolyphonyMax_2", "ACTrackNoteDensityMin_8"]),
    ],
)


@pytest.mark.parametrize(
    "tokenizer", [MMM(params=Path(__file__).parent.parent / "runs" / "tokenizer.json")]
)
@pytest.mark.parametrize("inference_config", [INFERENCE_CONFIG])
@pytest.mark.parametrize("input_midi_path", MIDI_PATHS)
def test_generate(
    tokenizer: MMM, inference_config: InferenceConfig, input_midi_path: str | Path
):
    model = DummyModel(tokenizer)

    _ = generate(
        model,
        tokenizer,
        inference_config,
        input_midi_path,
    )

    """output_score.dump_midi(
        Path(__file__).parent / "tests_output" / "midi_out_bpe_dummy.mid"
    )"""
