"""Lists the Experiment baselines and training."""

from __future__ import annotations

from random import choice, choices, uniform
from typing import TYPE_CHECKING

from miditok.attribute_controls import create_random_ac_indexes
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.data_augmentation.data_augmentation import (
    _filter_offset_tuples_to_score,
    augment_score,
)
from miditok.pytorch_data import DatasetMIDI
from miditok.utils import (
    get_average_num_tokens_per_note,
    split_score_per_note_density,
)
from symusic import Score
from torch import LongTensor

from utils.constants import MAX_NUM_FILES_NUM_TOKENS_PER_NOTE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from miditok import MMM, TokSequence


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
    :param pre_tokenize: whether to pre-tokenize the data files when creating the
        ``Dataset`` object. If this is enabled, the ``Dataset`` will tokenize all the
        files at its initialization and store the tokens in memory.
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
        pre_tokenize: bool = False,
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

        super().__init__(
            files_paths,
            tokenizer,
            max_seq_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pre_tokenize=pre_tokenize,
            ac_tracks_random_ratio_range=ac_tracks_random_ratio_range,
            ac_bars_random_ratio_range=ac_bars_random_ratio_range,
            func_to_get_labels=func_to_get_labels,
            sample_key_name=sample_key_name,
            labels_key_name=labels_key_name,
        )

    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """
        Return the ``idx`` elements of the dataset.

        If the dataset is pre-tokenized, the method will return the token ids.
        Otherwise, it will tokenize the ``idx``th file on the fly. If the file to is
        corrupted, the method will return an dictionary with ``None`` values.

        :param idx: idx of the file/sample.
        :return: the token ids, with optionally the associated label.
        """
        # Tokenize on the fly
        try:
            score = Score(self.files_paths[idx])
        except SCORE_LOADING_EXCEPTION:  # shouldn't happen if the dataset is cleaned
            return {self.sample_key_name: None}

        # We preprocess the Score here, before selecting the tracks to keep as some may
        # have been deleted.
        score = self.tokenizer.preprocess_score(score)

        # Select k tracks (shuffled)
        num_tracks_to_keep = round(
            len(score.tracks) * uniform(*self.ratio_random_tracks_range)  # noqa: S311
        )
        tracks = choices(score.tracks, k=num_tracks_to_keep)  # noqa: S311
        score.tracks = tracks

        # Augment the Score with randomly selected offsets among possible ones
        pitch_offsets = _filter_offset_tuples_to_score(
            self.pitch_offsets.copy(),
            score,
            restrict_on_program_tessitura=True,
        )
        score = augment_score(
            score,
            choice(pitch_offsets),  # noqa: S311
            choice(self.velocity_offsets),  # noqa: S311
            choice(self.duration_offsets),  # noqa: S311
        )

        # Select specific chunk of about x tokens
        score_chunks = split_score_per_note_density(
            score,
            self.max_seq_len,
            self.average_num_tokens_per_note,
            num_overlap_bars=0,
        )  # TODO make sure most make no more than max_seq_len
        score = choice(score_chunks)  # noqa: S311

        # TODO Place infilling tokens on randomly selected tracks/bars
        """track_fill = random() > self.bar_fill_ratio
        if track_fill:
            t = 0
        else:
            bars_ticks = get_bars_ticks(score)
            # TODO inpaint on n tracks randomly selected
            bars_section = 0"""

        # TODO External labels: (non-)expressive, loops, genres

        # Tokenize
        tokseq = self._tokenize_score(score)
        # If not one_token_stream, we only take the first track/sequence
        token_ids = tokseq.ids if self.tokenizer.one_token_stream else tokseq[0].ids

        return {self.sample_key_name: LongTensor(token_ids)}

    def _tokenize_score(self, score: Score) -> TokSequence:
        # Preprocess the music file
        score = self.tokenizer.preprocess_score(score)

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
        tokseq = self.tokenizer.encode(
            score,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
            concatenate_track_sequences=False,  # TODO select track infill sequence here
        )

        # If tokenizing on the fly a multi-stream tokenizer, only keeps the first track
        if not self.pre_tokenize and not self.tokenizer.one_token_stream:
            tokseq = [tokseq[0]]

        # If this file is a chunk (split_files_for_training), determine its id.
        # By default, we add BOS and EOS tokens following the values of
        # self.bos_token_id and self.eos_token_id (that may be None), except when the
        # file is identified as a chunk.
        add_bos_token = add_eos_token = True
        for marker in score.markers:
            if marker.time != 0:
                break
            if marker.text.startswith("miditok: chunk"):
                chunk_id, chunk_id_last = map(
                    int, marker.text.split(" ")[-1].split("/")
                )
                add_bos_token = chunk_id == 0
                add_eos_token = chunk_id == chunk_id_last

        # Preprocessing token ids: reduce sequence length, add BOS/EOS tokens
        tokseq = sum(tokseq)
        tokseq.ids = self._preprocess_token_ids(
            tokseq.ids,
            self.max_seq_len,
            self.bos_token_id if add_bos_token else None,
            self.eos_token_id if add_eos_token else None,
            enforce_eos_token_if_seq_len_exceed_lim=False,
        )
        return tokseq
