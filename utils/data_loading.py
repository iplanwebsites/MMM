"""Lists the Experiment baselines and training."""

from __future__ import annotations

from random import choices, shuffle, uniform
from typing import TYPE_CHECKING

from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.pytorch_data import DatasetMIDI
from miditok.utils import get_bars_ticks
from symusic import Score
from torch import LongTensor

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from miditok import TokSequence
    from tokentamer import Controller


class DatasetMMM(DatasetMIDI):
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
    :param controller: TokenTamer Controller.
    :param max_seq_len: maximum sequence length (in num of tokens)
    :param ratio_random_tracks_range:
    :param tracks_idx_random_ratio_range: range of ratios (between 0 and 1 included) of
        tracks to compute attribute controls on.
    :param bars_idx_random_ratio_range: range of ratios (between 0 and 1 included) of
        bars to compute attribute controls on.
    :param acs_idx_random_ratio_range: range of ratios (between 0 and 1 included) of
        attribute controls to compute for each selected tracks and bars.
    :param bos_token_id: *BOS* token id. (default: ``None``)
    :param eos_token_id: *EOS* token id. (default: ``None``)
    :param sample_key_name: name of the dictionary key containing the sample data when
        iterating the dataset. (default: ``"input_ids"``)
    :param labels_key_name: name of the dictionary key containing the labels data when
        iterating the dataset. (default: ``"labels"``)
    """

    def __init__(
        self,
        files_paths: Sequence[Path],
        controller: Controller,
        max_seq_len: int,
        ratio_random_tracks_range: tuple[float, float],
        acs_idx_random_ratio_range: tuple[float, float],
        tracks_idx_random_ratio_range: tuple[float, float],
        bars_idx_random_ratio_range: tuple[float, float],
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        sample_key_name: str = "input_ids",
        labels_key_name: str = "labels",
    ) -> None:
        self.controller = controller
        self.ratio_random_tracks_range = ratio_random_tracks_range
        self.acs_idx_random_ratio_range = acs_idx_random_ratio_range
        self.tracks_idx_random_ratio_range = tracks_idx_random_ratio_range
        self.bars_idx_random_ratio_range = bars_idx_random_ratio_range
        super().__init__(
            files_paths,
            self.controller.tokenizer,
            max_seq_len,
            pre_tokenize=False,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            func_to_get_labels=None,
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
        except SCORE_LOADING_EXCEPTION:
            return {self.sample_key_name: None}

        # We preprocess the Score here, before selecting the tracks to keep as some may
        # have been deleted.
        score = self.controller.tokenizer.preprocess_score(score)

        # Select k tracks
        num_tracks_to_keep = round(
            len(score.tracks) * uniform(*self.ratio_random_tracks_range)  # noqa: S311
        )
        tracks = choices(score.tracks, k=num_tracks_to_keep)  # noqa: S311
        shuffle(tracks)
        score.tracks = tracks

        # TODO place infilling tokens on randomly selected tracks/bars
        # TODO (non-)expressive, loops, genres

        tokseq = self._tokenize_score(score)
        # If not one_token_stream, we only take the first track/sequence
        token_ids = tokseq.ids if self.tokenizer.one_token_stream else tokseq[0].ids

        return {self.sample_key_name: LongTensor(token_ids)}

    def _tokenize_score(self, score: Score) -> TokSequence:
        # Get random idx of ACs, tracks and bars
        num_acs_controller = len(self.controller.attribute_controls)
        num_acs = round(
            num_acs_controller * uniform(*self.acs_idx_random_ratio_range)  # noqa: S311
        )
        acs_idx = choices(list(range(num_acs_controller)), k=num_acs)  # noqa: S311
        num_tracks = round(
            len(score.tracks) * uniform(*self.tracks_idx_random_ratio_range)  # noqa: S311
        )
        tracks_idx = choices(list(range(len(score.tracks))), k=num_tracks)  # noqa: S311
        num_bars_score = len(get_bars_ticks(score))
        num_bars = round(
            num_bars_score * uniform(*self.bars_idx_random_ratio_range)  # noqa: S311
        )
        bars_idx = choices(list(range(num_bars_score)), k=num_bars)  # noqa: S311

        # Inject attribute controls
        self.controller.inject_control_tokens(
            tokseq := self.tokenizer(
                score, no_preprocess_score=True, concatenate_track_sequences=False
            ),
            score,
            acs_idx,
            tracks_idx,
            bars_idx,
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
