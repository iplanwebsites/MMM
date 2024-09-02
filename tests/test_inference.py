"""Tests for MMM inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from miditok import MMM, TokenizerConfig
from transformers import MistralConfig, MistralForCausalLM
from .utils_tests import MIDI_PATH

from mmm.inference import generate
from utils.classes import InferenceConfig
from utils.constants import (
    SLIDING_WINDOWS,
    TOKENIZER_PARAMS,
)

if TYPE_CHECKING:
    from pathlib import Path

# TODO: Test track generation
INFERENCE_CONFIG = InferenceConfig(
    {
        0: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
        2: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
        3: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
    },
    []
    #[
    #    (43, ["ACTrackOnsetPolyphonyMax_2", "ACTrackNoteDensityMin_8"]),
    #],
)

MISTRAL_CONFIG = MistralConfig(
    hidden_size=128,
    intermediate_size=128 * 4,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    sliding_window=SLIDING_WINDOWS,
)


@pytest.mark.parametrize("tokenizer", [MMM(TokenizerConfig(**TOKENIZER_PARAMS))])
@pytest.mark.parametrize("inference_config", [INFERENCE_CONFIG])
@pytest.mark.parametrize("input_midi_path", [MIDI_PATH])
def test_generate(
    tokenizer: MMM, inference_config: InferenceConfig, input_midi_path: str | Path
):
    MISTRAL_CONFIG.vocab_size = tokenizer.vocab_size
    output_score = generate(
        MistralForCausalLM(MISTRAL_CONFIG),
        tokenizer,
        inference_config,
        input_midi_path,
    )

    output_score.dump_midi("tests_output/midi_out.mid")


