from pathlib import Path

import pytest
from transformers import MistralForCausalLM, MistralConfig
from miditok import TokenizerConfig, MMM

from utils.classes import InferenceConfig
from mmm.inference import generate

from utils_tests import MIDI_PATH
from utils.constants import (
    TOKENIZER_PARAMS,
    MAX_POSITION_EMBEDDINGS,
    EMBEDDING_SIZE,
    FEEDFORWARD_SIZE,
    NUM_LAYERS,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    SLIDING_WINDOWS
)

INFERENCE_CONFIG = InferenceConfig(
    {
        0: [(12,16, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
        3: [(24,26, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])]
    },
    [
        (43, ["ACTrackOnsetPolyphonyMax_2", "ACTrackNoteDensityMin_8"]),
    ]

)

MISTRAL_CONFIG = MistralConfig(
    hidden_size=128,
    intermediate_size=128*4,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    sliding_window=SLIDING_WINDOWS
)


@pytest.mark.parametrize(
    "test_input,expected",
    [("3+5", 8), ("2+4", 6), pytest.param("6*9", 42, marks=pytest.mark.xfail)],
)
def test_eval(test_input, expected):
    assert eval(test_input) == expected


@pytest.mark.parametrize("tokenizer", [MMM(TokenizerConfig(**TOKENIZER_PARAMS))])
@pytest.mark.parametrize("inference_config", [INFERENCE_CONFIG])
@pytest.mark.parametrize("input_midi_path", [MIDI_PATH])
def test_generate(
    tokenizer: MMM,
    inference_config: InferenceConfig,
    input_midi_path: str | Path
):
    output_path = "./midis/generated_midi"
    MISTRAL_CONFIG.vocab_size = tokenizer.vocab_size
    generate(MistralForCausalLM(MISTRAL_CONFIG), tokenizer, inference_config, input_midi_path, output_path)
