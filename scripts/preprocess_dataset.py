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

    from miditok import TokTrainingIterator
    from miditok.constants import SCORE_LOADING_EXCEPTION
    from transformers.trainer_utils import set_seed

    from mmm import mmm
    from utils.constants import (
        BARS_IDX_RANDOM_RATIO_RANGE,
        TRACKS_IDX_RANDOM_RATIO_RANGE,
        TRAINING_MAX_NUM_FILES,
    )

    set_seed(mmm.seed)

    # Filter non-valid files
    dataset_files_paths = mmm.dataset_files_paths
    for file_path in dataset_files_paths:
        try:
            score_ = Score(file_path)
        except SCORE_LOADING_EXCEPTION:
            file_path.unlink()
            continue
        if not is_score_valid(score_, MIN_NUM_BARS_FILE_VALID):
            file_path.unlink()

    # Train the tokenizer
    dataset_files_paths = mmm.dataset_files_paths
    shuffle(dataset_files_paths)
    dataset_files_paths_training = dataset_files_paths[:TRAINING_MAX_NUM_FILES]
    iterator = TokTrainingIterator(
        mmm.tokenizer,
        dataset_files_paths_training,
        TRACKS_IDX_RANDOM_RATIO_RANGE,
        BARS_IDX_RANDOM_RATIO_RANGE,
    )
    mmm.tokenizer.train(
        vocab_size=mmm.tokenization_config.vocab_size,
        iterator=iterator,
    )
    mmm.tokenizer.save(mmm.tokenizer_path)
