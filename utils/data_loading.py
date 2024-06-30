"""Lists the Experiment baselines and training."""

from __future__ import annotations

from random import choice, random, sample, uniform
from typing import TYPE_CHECKING

import numpy as np
from miditok.attribute_controls import create_random_ac_indexes
from miditok.data_augmentation.data_augmentation import (
    _filter_offset_tuples_to_score,
    augment_score,
)
from miditok.pytorch_data import DatasetMIDI
from miditok.utils import (
    get_average_num_tokens_per_note,
    split_score_per_note_density,
)

from utils.constants import MAX_NUM_FILES_NUM_TOKENS_PER_NOTE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from miditok import MMM, TokSequence
    from symusic import Score
    from torch import LongTensor


class DatasetMMMPreTok(DatasetMIDI):
    r"""
    A ``Dataset`` loading and tokenizing music files (MIDI, abc) during training.

    This class can be used for either tokenize music files on the fly when iterating it,
    or by pre-tokenizing all the files at its initialization and store the tokens in
    memory.

    **Important note:** you should probably use this class in concert with the
    :py:func:`miditok.pytorch_data.split_files_for_training` method in order to train
    your model with chunks of music files having token sequence lengths close to the
    ``max_seq_len`` value.
    When using this class with file chunks, the ``BOS`` and ``EOS`` tokens will only be
    added to the first and last chunks respectively. This allows to not train the model
    with ``EOS`` tokens that would incorrectly inform the model the end of the data
    samples, and break the causality chain of consecutive chunks with incorrectly
    placed ``BOS`` tokens.

    Additionally, you can use the ``func_to_get_labels`` argument to provide a method
    allowing to use labels (one label per file).

    **Handling of corrupted files:**
    Some MIDI files may be corrupted, as for example containing unexpected values.
    In such cases, if the ``DatasetMIDI`` pre-tokenizes, it will simply ignore these
    files. Otherwise, the ``DatasetMIDI`` will return dictionaries with ``None`` values
    when iterated.

    :param files_paths: paths to the music files to load.
    :param tokenizer: MidiTok tokenizer.
    :param max_seq_len: maximum sequence length (in num of tokens)
    :param ratio_random_tracks_range:
    :param data_augmentation_offsets: tuple of data augmentation offsets for pitch,
        velocity and duration values.
    :param bar_fill_ratio: ratio between 0 and 1 at which a sample should be
        "bar-filled". The difference between 1 and this ratio accounts for the ratio for
        which a sample will be "track-filled".
    :param bar_masking_ratio_range: range of the random ratio of bars to mask during
        training when infilling bars.
    :param bos_token_id: *BOS* token id. (default: ``None``)
    :param eos_token_id: *EOS* token id. (default: ``None``)
    :param ac_tracks_random_ratio_range: range of ratios (between 0 and 1 included) of
        tracks to compute attribute controls on. If ``None`` is given, the attribute
        controls will be computed for all the tracks. (default: ``None``)
    :param ac_bars_random_ratio_range: range of ratios (between 0 and 1 included) of
        bars to compute attribute controls on. If ``None`` is given, the attribute
        controls will be computed for all the bars. (default: ``None``)
    :param sample_key_name: name of the dictionary key containing the sample data when
        iterating the dataset. (default: ``"input_ids"``)
    :param labels_key_name: name of the dictionary key containing the labels data when
        iterating the dataset. (default: ``"labels"``)
    """

    def __init__(
        self,
        files_paths: Sequence[Path],
        tokenizer: MMM,
        max_seq_len: int,
        ratio_random_tracks_range: tuple[float, float],
        data_augmentation_offsets: tuple[int, int, int],
        bar_fill_ratio: float,
        bar_masking_ratio_range: tuple[float, float],
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        ac_tracks_random_ratio_range: tuple[float, float] | None = None,
        ac_bars_random_ratio_range: tuple[float, float] | None = None,
        func_to_get_labels: Callable[
            [Score, TokSequence | list[TokSequence], Path],
            int | list[int] | LongTensor,
        ]
        | None = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ) -> None:
        self.ratio_random_tracks_range = ratio_random_tracks_range
        pitch_offsets = data_augmentation_offsets[0]
        self.pitch_offsets = list(range(-pitch_offsets, pitch_offsets + 1))
        velocity_offsets = data_augmentation_offsets[1]
        self.velocity_offsets = list(range(-velocity_offsets, velocity_offsets + 1))
        duration_offsets = data_augmentation_offsets[2]
        self.duration_offsets = list(range(-duration_offsets, duration_offsets + 1))
        self.bar_fill_ratio = bar_fill_ratio
        self.bar_masking_ratio_range = bar_masking_ratio_range

        self.average_num_tokens_per_note = get_average_num_tokens_per_note(
            self.tokenizer, files_paths[:MAX_NUM_FILES_NUM_TOKENS_PER_NOTE]
        )

        # Infill tokens, set as attribute here to avoid to access to vocab dic
        self._infill_bar_token = "InFill_Bar"
        self._infill_track_start_token = "FillTrack_Start"
        self._infill_track_end_token = "FillTrack_End"
        self._infill_bar_start_token = "FillBar_Start"
        self._infill_bar_end_token = "FillBar_End"
        self._infill_bar_token_id = self.tokenizer.vocab["InFill_Bar"]
        self._infill_track_start_token_id = self.tokenizer.vocab["FillTrack_Start"]
        self._infill_track_end_token_id = self.tokenizer.vocab["FillTrack_End"]
        self._infill_bar_start_token_id = self.tokenizer.vocab["FillBar_Start"]
        self._infill_bar_end_token_id = self.tokenizer.vocab["FillBar_End"]

        super().__init__(
            files_paths,
            tokenizer,
            max_seq_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pre_tokenize=False,  # can't pre-tokenize with the random selections
            ac_tracks_random_ratio_range=ac_tracks_random_ratio_range,
            ac_bars_random_ratio_range=ac_bars_random_ratio_range,
            func_to_get_labels=func_to_get_labels,
            sample_key_name=sample_key_name,
            labels_key_name=labels_key_name,
        )

    def _tokenize_score(self, score: Score) -> TokSequence:
        # Preprocess the music file
        score = self.tokenizer.preprocess_score(score)

        # Select k tracks (shuffled)
        num_tracks_to_keep = round(
            len(score.tracks) * uniform(*self.ratio_random_tracks_range)
        )
        tracks = sample(score.tracks, k=num_tracks_to_keep)
        score.tracks = tracks

        # Augment the Score with randomly selected offsets among possible ones
        pitch_offsets = _filter_offset_tuples_to_score(
            self.pitch_offsets.copy(),
            score,
            restrict_on_program_tessitura=True,
        )
        score = augment_score(
            score,
            choice(pitch_offsets),
            choice(self.velocity_offsets),
            choice(self.duration_offsets),
        )

        # Select specific chunk of about x tokens
        score_chunks = split_score_per_note_density(
            score,
            self.max_seq_len,
            self.average_num_tokens_per_note,
            num_overlap_bars=0,
        )  # TODO make sure most make no more than max_seq_len
        score = choice(score_chunks)

        # Randomly create attribute controls indexes
        ac_indexes = None
        if self.tracks_idx_random_ratio_range or self.bars_idx_random_ratio_range:
            ac_indexes = create_random_ac_indexes(
                score,
                self.tokenizer.attribute_controls,
                self.tracks_idx_random_ratio_range,
                self.bars_idx_random_ratio_range,
            )

        # Tokenize it
        sequences = self.tokenizer.encode(
            score,
            encode_ids=False,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
            concatenate_track_sequences=False,
        )

        # Place infilling tokens on randomly selected tracks/bars
        if random() > self.bar_fill_ratio:
            # Add Track Fill start/end tokens
            seq_infill = sequences.pop(choice(list(range(len(sequences)))))
            seq_infill.ids.insert(0, self._infill_track_start_token_id)
            seq_infill.ids.append(self._infill_track_end_token_id)
            sequences.append(seq_infill)
        else:
            # Bar infilling
            # Determine the portion to infill
            bars_ticks = sequences[0]._ticks_bars
            bar_section_length = max(
                1,
                round(len(bars_ticks) * uniform(*self.bar_masking_ratio_range)),
            )
            bar_idx_start = choice(range(len(bars_ticks) - bar_section_length))
            bar_idx_end = bar_idx_start + bar_section_length
            bar_tick_start = bars_ticks[bar_idx_start]
            bar_tick_end = bars_ticks[bar_idx_end]

            # Extract token sections of the bars to infill for each track/seq
            extracted_seqs = []
            for si in range(len(sequences)):
                times = np.array([event.time for event in sequences[si].events])
                token_idx_start = np.nonzero(times <= bar_tick_start)[0]
                token_idx_end = np.nonzero(times >= bar_tick_end)[0]

                # Add track start/end + Program for the extracted bars sections
                ti = 0
                while sequences[si].events[ti].type_ != "Program":
                    ti += 1
                program = sequences[si].events[ti].value
                seq_infill = sequences[si][token_idx_start:token_idx_end]
                seq_infill.ids.insert(0, self.tokenizer.vocab[f"Program_{program}"])
                seq_infill.tokens.insert(0, f"Program_{program}")
                seq_infill.ids.insert(0, self.tokenizer.vocab["Track_Start"])
                seq_infill.tokens.insert(0, "Track_Start")
                seq_infill.ids.append(self.tokenizer.vocab["Track_End"])
                seq_infill.tokens.append("Track_End")
                extracted_seqs.append(seq_infill)

                # Add BarInfill tokens + update sequences
                seq_before = sequences[si][:token_idx_start]
                seq_before.ids.append(self._infill_bar_token_id)
                seq_before.tokens.append(self._infill_bar_token)
                seq_after = sequences[si][token_idx_end:]
                sequences[si] = seq_before + seq_after

            # Add BarFill start/end
            extracted_seqs = sum(extracted_seqs)
            extracted_seqs.ids.insert(0, self._infill_bar_start_token_id)
            extracted_seqs.tokens.insert(0, self._infill_bar_start_token)
            extracted_seqs.ids.append(self._infill_bar_end_token_id)
            extracted_seqs.tokens.append(self._infill_bar_end_token)
            sequences.append(extracted_seqs)

        # TODO External labels: (non-)expressive, loops, genres to add to the seq

        # Preprocessing token ids: reduce sequence length, add BOS/EOS tokens
        tokseq = sum(sequences)
        self.tokenizer.encode_token_ids(tokseq)
        tokseq.ids = self._preprocess_token_ids(
            tokseq.ids,
            self.max_seq_len,
            self.bos_token_id,
            self.eos_token_id,
            enforce_eos_token_if_seq_len_exceed_lim=False,
        )
        return tokseq
