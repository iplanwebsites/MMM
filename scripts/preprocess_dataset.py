#!/usr/bin/python3 python

"""Script deleting non-valid MIDI files from the dataset and training the tokenizer."""

from miditok.utils import get_bars_ticks
from symusic import Score

from utils.constants import MIN_NUM_BARS_FILE_VALID


def is_score_valid(score: Score, min_num_bars: int = MIN_NUM_BARS_FILE_VALID) -> bool:
    """
    Check if a ``symusic.Score`` is valid, contains the minimum required number of bars.

    :param score: ``symusic.Score`` to inspect.
    :param min_num_bars: minimum number of bars the score should contain.
    :return: boolean indicating if ``score`` is valid.
    """
    return len(get_bars_ticks(score)) >= min_num_bars


if __name__ == "__main__":
    from random import shuffle

    from miditok.constants import SCORE_LOADING_EXCEPTION
    from tokentamer.tokenizer_training_iterator import TokTrainingIterator
    from transformers.trainer_utils import set_seed

    from mmm import mmm
    from utils.constants import (
        ACS_RANDOM_RATIO_RANGE,
        BARS_IDX_RANDOM_RATIO_RANGE,
        TRACKS_IDX_RANDOM_RATIO_RANGE,
        TRAINING_MAX_NUM_FILES,
    )

    set_seed(mmm.seed)

    # Filter non-valid files
    dataset_files_paths = mmm.dataset_files_paths
    for idx in range(len(dataset_files_paths), -1, -1):
        try:
            score_ = Score(dataset_files_paths[idx])
        except SCORE_LOADING_EXCEPTION:
            del dataset_files_paths[idx]
            continue

        if not is_score_valid(score_, MIN_NUM_BARS_FILE_VALID):
            del dataset_files_paths[idx]

    # Train the tokenizer
    dataset_files_paths = mmm.dataset_files_paths
    shuffle(dataset_files_paths)
    dataset_files_paths_training = dataset_files_paths[:TRAINING_MAX_NUM_FILES]
    iterator = TokTrainingIterator(
        mmm.controller,
        dataset_files_paths_training,
        ACS_RANDOM_RATIO_RANGE,
        TRACKS_IDX_RANDOM_RATIO_RANGE,
        BARS_IDX_RANDOM_RATIO_RANGE,
    )
    mmm.tokenizer.train(
        vocab_size=mmm.tokenization_config.vocab_size,
        iterator=iterator,
    )
    mmm.tokenizer.save(mmm.tokenizer_path)
