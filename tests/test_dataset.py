from pathlib import Path
import glob, os
from utils.data_loading import DatasetMMM

import pytest
from miditok import MMM, TokenizerConfig

from datasets import load_dataset

from utils.constants import (
    TOKENIZER_PARAMS,
    TRACKS_SELECTION_RANDOM_RATIO_RANGE,
    RATIO_BAR_INFILLING,
    RATIOS_RANGE_BAR_INFILLING_DURATION,
    DATA_AUGMENTATION_OFFSETS,
    MAX_SEQ_LEN,
TRACKS_IDX_RANDOM_RATIO_RANGE,
ACS_RANDOM_RATIO_RANGE,
BARS_IDX_RANDOM_RATIO_RANGE
)

import huggingface_hub

from symusic import Score


@pytest.mark.parametrize("tokenizer", [MMM(params="C:/Users/rizzo/OneDrive/Desktop/TESI/MMM/runs/MMM_Mistral/tokenizer.json")])
def test_dataset(tokenizer: MMM):
    huggingface_hub.login(token="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")

    dataset = load_dataset("Metacreation/GigaMIDI", "all-instruments-with-drums", trust_remote_code=True, streaming=True,
                           use_auth_token="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")
    dataset_obj = DatasetMMM(dataset, tokenizer,
               MAX_SEQ_LEN,
               TRACKS_SELECTION_RANDOM_RATIO_RANGE,
                DATA_AUGMENTATION_OFFSETS,
                RATIO_BAR_INFILLING,
                RATIOS_RANGE_BAR_INFILLING_DURATION,
                 ac_random_ratio_range=ACS_RANDOM_RATIO_RANGE,
                 ac_tracks_random_ratio_range=TRACKS_IDX_RANDOM_RATIO_RANGE,
                 ac_bars_random_ratio_range=BARS_IDX_RANDOM_RATIO_RANGE,
               )

    for filepath in glob.glob("midis/*"):
        if os.path.isfile(filepath):
            print(f"Analyzing file: {os.path.basename(filepath)}")
        score = Score(filepath)
        #score = dataset_obj.augment_and_preprocess_score(score)
        tokseq, decoder_input = dataset_obj._tokenize_score(score)
        print(tokseq)

    print("DONE")


