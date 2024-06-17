#!/usr/bin/python3 python

"""Script to transform the MIDI datasets into tokens."""

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

    from miditok.data_augmentation import augment_dataset
    from miditok.utils import split_files_for_training
    from tokentamer.tokenizer_training_iterator import TokTrainingIterator
    from transformers.trainer_utils import set_seed

    from mmm import mmm
    from utils.constants import (
        ACS_RANDOM_RATIO_RANGE,
        BARS_IDX_RANDOM_RATIO_RANGE,
        DATA_CHUNK_NUM_OVERLAP_BARS,
        TRACKS_IDX_RANDOM_RATIO_RANGE,
        TRAINING_MAX_NUM_FILES,
    )

    set_seed(mmm.seed)

    # Train the tokenizer-
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

    # Split MIDI paths in train/valid/test sets
    total_num_files = len(dataset_files_paths)
    num_files_valid = round(total_num_files * mmm.data_config.ratio_valid_subset)
    num_files_test = round(total_num_files * mmm.data_config.ratio_test_subset)
    midi_paths_valid = dataset_files_paths[:num_files_valid]
    midi_paths_test = dataset_files_paths[
        num_files_valid : num_files_valid + num_files_test
    ]
    midi_paths_train = dataset_files_paths[num_files_valid + num_files_test :]

    # Chunk MIDIs and perform data augmentation on each subset independently
    subsets_dir_paths = mmm.data_subsets_paths
    pitch_offsets = mmm.data_config.data_augmentation_offsets[0]
    pitch_offsets = list(range(-pitch_offsets, pitch_offsets + 1))
    velocity_offsets = mmm.data_config.data_augmentation_offsets[1]
    velocity_offsets = list(range(-velocity_offsets, velocity_offsets + 1))
    duration_offsets = mmm.data_config.data_augmentation_offsets[2]
    duration_offsets = list(range(-duration_offsets, duration_offsets + 1))
    for files_paths, subset_path in zip(
        (midi_paths_train, midi_paths_valid, midi_paths_test), mmm.data_subsets_paths
    ):
        # Split the MIDIs into chunks of sizes approximately `max_seq_len` tokens
        split_files_for_training(
            files_paths=files_paths,
            tokenizer=mmm.tokenizer,
            save_dir=subset_path,
            max_seq_len=mmm.data_config.max_seq_len,
            num_overlap_bars=DATA_CHUNK_NUM_OVERLAP_BARS,
        )

        # Perform data augmentation
        augment_dataset(
            subset_path,
            pitch_offsets=pitch_offsets,
            velocity_offsets=velocity_offsets,
            duration_offsets=duration_offsets,
        )
