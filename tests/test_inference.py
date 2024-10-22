"""Tests for MMM inference."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from miditok import MMM
from transformers import GenerationConfig, MistralForCausalLM

from mmm import InferenceConfig, StopLogitsProcessor, generate
from scripts.utils.constants import (
    EPSILON_CUTOFF,
    ETA_CUTOFF,
    MAX_LENGTH,
    NUM_BEAMS,
    REPETITION_PENALTY,
    TEMPERATURE_SAMPLING,
    TOP_K,
    TOP_P,
)

from .utils_tests import MIDI_PATH

INFERENCE_CONFIG = InferenceConfig(
    {
        #0: [(14, 18, []), (30, 34, [])],
        #0: [(14, 18, [])],
        #1: [(40, 44, [])],
        # 2: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
        # 3: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
    },
    [
        (43, []),
    ],
)


@pytest.mark.parametrize(
    "tokenizer", [MMM(params=Path(__file__).parent.parent / "runs" / "tokenizer.json")]
)
@pytest.mark.parametrize("inference_config", [INFERENCE_CONFIG])
@pytest.mark.parametrize("input_midi_path", MIDI_PATH)
def test_generate(
    tokenizer: MMM, inference_config: InferenceConfig, input_midi_path: str | Path
):
    model = MistralForCausalLM.from_pretrained(Path(__file__).parent.parent
                                               / "models" / "checkpoint-33000",
                                               use_safetensors=True)
    logits_processor = StopLogitsProcessor(tokenizer.vocab["Bar_None"],
                                           tokenizer.vocab["FillBar_End"], tokenizer)

    gen_config = GenerationConfig(
        num_beams=NUM_BEAMS,
        temperature=TEMPERATURE_SAMPLING,
        repetition_penalty=REPETITION_PENALTY,
        top_k=TOP_K,
        top_p=TOP_P,
        epsilon_cutoff=EPSILON_CUTOFF,
        eta_cutoff=ETA_CUTOFF,
        max_length=MAX_LENGTH
    )

    _ = generate(
        model,
        tokenizer,
        inference_config,
        input_midi_path,
        logits_processor,
        {
            "generation_config": gen_config
        }
    )

    t = time.localtime()
    filename = Path(input_midi_path).stem
    _.dump_midi(
        Path(__file__).parent / "tests_output" / "33kT12" /
        f"{filename}{t.tm_min}{t.tm_sec}.mid"
    )
