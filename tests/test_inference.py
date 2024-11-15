"""Tests for MMM inference."""

from __future__ import annotations

import time
from pathlib import Path

import os
import pytest
import symusic
from miditok import MMM
from transformers import GenerationConfig, MistralForCausalLM
import random
from datetime import datetime
import json

import numpy as np

from mmm import InferenceConfig, StopLogitsProcessor, generate
from scripts.utils.constants import (
    EPSILON_CUTOFF,
    ETA_CUTOFF,
    MAX_NEW_TOKENS,
    NUM_BEAMS,
    REPETITION_PENALTY,
    TEMPERATURE_SAMPLING,
    TOP_K,
    TOP_P,
    MAX_LENGTH
)

from .utils_tests import MIDI_PATH

INFERENCE_CONFIG = InferenceConfig(
    {
        # 0: [(14, 18, []), (30, 34, [])],
        # 0: [(14, 18, [])],
        1: [(40, 44, [])],
        # 2: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
        # 3: [(4, 8, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
    },
    []
    #[
    #    (43, []),
    #],
)


@pytest.mark.parametrize(
    "tokenizer", [MMM(params=Path(__file__).parent.parent / "runs" / "tokenizer.json")]
)
@pytest.mark.parametrize("inference_config", [INFERENCE_CONFIG])
@pytest.mark.parametrize("input_midi_path", MIDI_PATH)
#@pytest.mark.skip(reason="no way of currently testing this")
def test_generate(
    tokenizer: MMM, inference_config: InferenceConfig, input_midi_path: str | Path
):
    model = MistralForCausalLM.from_pretrained(
        Path(__file__).parent.parent / "models" / "checkpoint-87000",
        use_safetensors=True,
    )



    gen_config = GenerationConfig(
        num_beams=NUM_BEAMS,
        temperature=TEMPERATURE_SAMPLING,
        repetition_penalty=REPETITION_PENALTY,
        top_k=TOP_K,
        top_p=TOP_P,
        epsilon_cutoff=EPSILON_CUTOFF,
        eta_cutoff=ETA_CUTOFF,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length = MAX_LENGTH,
    )

    # Get number of tracks and number of bars of the MIDI track
    score = symusic.Score(input_midi_path)
    tokens = tokenizer.encode(score, concatenate_track_sequences=False)
    num_bars = len(tokens[0]._ticks_bars)
    num_tracks = len(tokens)

    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    output_folder_path = Path(__file__).parent / "tests_output" / "87k" / f"TEST_1"/f"test_{str(input_midi_path.name)}"

    os.makedirs(output_folder_path, exist_ok=True)

    # Write gen_config to JSON file
    gen_config_dict = vars(gen_config)

    i = 0
    # To be added to JSON file
    infillings = []
    while i < 11:
        bar_idx_infill_start = random.randint(16, num_bars - 17)
        track_idx = random.randint(0, num_tracks - 1)

        bars_ticks = tokens[track_idx]._ticks_bars
        bar_tick_end = bars_ticks[bar_idx_infill_start + 8]

        times = np.array([event.time for event in tokens[track_idx].events])
        token_idx_end = np.nonzero(times >= bar_tick_end)[0]

        if len(token_idx_end) == 0:
            print(
                f"Ignoring infilling of bars {bar_idx_infill_start} - {bar_idx_infill_start + 4} on track {track_idx}")
            continue

        inference_config = InferenceConfig(
            {
                track_idx: [(bar_idx_infill_start, bar_idx_infill_start + 4, [])],
                #8: [(51, 55, [])],
            },
            []
        )

        entry = {
            "name": str(input_midi_path.name),
            "track_idx": track_idx,
            "start_bar_idx": bar_idx_infill_start,
            "end_bar_idx": bar_idx_infill_start + 4,
        }

        j = 0
        while j < 1:

            start_time = time.time()

            _ = generate(
                model,
                tokenizer,
                inference_config,
                input_midi_path,
                {"generation_config": gen_config},
            )

            end_time = time.time()

            #filename = Path(input_midi_path).stem
            _.dump_midi(
                output_folder_path
                / f"track{track_idx}_infill_bars{bar_idx_infill_start}_{bar_idx_infill_start+4}_generationtime_{end_time - start_time}.midi.mid"
            )

            j+=1

        infillings.append(entry)

        i += 1



    json_data = {
        "generation_config": gen_config_dict,
        "infillings": infillings
    }

    json_string = json.dumps(json_data, indent=4)
    with open(f"{output_folder_path}/generation_config.json", "w") as file:
        file.write(json_string)

