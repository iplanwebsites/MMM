"""Lists the Experiment baselines and training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from random import choice, random, sample, shuffle, uniform
from typing import TYPE_CHECKING

import numpy as np
import torch
from miditok import TokSequence
from miditok.attribute_controls import BarAttributeControl
from miditok.constants import SCORE_LOADING_EXCEPTION
from miditok.data_augmentation.data_augmentation import (
    _filter_offset_tuples_to_score,
    augment_score,
)
from miditok.pytorch_data import DatasetMIDI
from miditok.utils import (
    get_average_num_tokens_per_note,
    get_bars_ticks,
    split_score_per_note_density,
)
from symusic import Score
from torch import LongTensor

from utils.constants import MAX_NUM_FILES_NUM_TOKENS_PER_NOTE, MIN_SEQ_LEN

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset
    from miditok import MMM


def concat_tokseq(sequences: list[TokSequence]) -> TokSequence:
    """
    Concatenate a sequence of :class:`miditok.TokSequence`.

    :param sequences: :class:`miditok.TokSequence`s to concatenate.
    :return: the concatenated ``sequences``.
    """
    tokseq = sequences.pop(0)
    for seq in sequences:
        tokseq += seq
    return tokseq


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

    :param dataset: hugging face Dataset instance.
    :param tokenizer: MidiTok tokenizer.
    :param max_seq_len: maximum sequence length (in num of tokens)
    :param ratio_random_tracks_range:
    :param data_augmentation_offsets: tuple of data augmentation offsets for pitch,
        velocity and duration values.
    :param bar_fill_ratio: ratio between 0 and 1 at which a sample should be
        "bar-filled". The difference between 1 and this ratio accounts for the ratio for
        which a sample will be "track-filled".
    :param bar_masking_duration_ratio_range: range of the random ratio of bars to mask
        during training when infilling bars.
    :param bos_token_id: *BOS* token id. (default: ``None``)
    :param eos_token_id: *EOS* token id. (default: ``None``)
    :param ac_random_ratio_range: range of ratios of number of attribute controls to be
        used on the portion of bars or track to infill. (default: ``None``)
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
        dataset: Dataset,
        tokenizer: MMM,
        max_seq_len: int,
        ratio_random_tracks_range: tuple[float, float],
        data_augmentation_offsets: tuple[int, int, int],
        bar_fill_ratio: float,
        bar_masking_duration_ratio_range: tuple[float, float],
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        ac_random_ratio_range: tuple[float, float] | None = None,
        ac_tracks_random_ratio_range: tuple[float, float] | None = None,
        ac_bars_random_ratio_range: tuple[float, float] | None = None,
        func_to_get_labels: Callable[
            [Score, TokSequence | list[TokSequence], Path],
            int | list[int] | LongTensor,
        ]
        | None = None,
        sample_key_name: str = "input_ids",
        decoder_key_name: str = "decoder_input_ids",
        labels_key_name: str = "labels",
        seq2seq: bool = False,
    ) -> None:
        self._dataset = dataset
        self.ratio_random_tracks_range = ratio_random_tracks_range
        pitch_offsets = data_augmentation_offsets[0]
        self.pitch_offsets = list(range(-pitch_offsets, pitch_offsets + 1))
        velocity_offsets = data_augmentation_offsets[1]
        self.velocity_offsets = list(range(-velocity_offsets, velocity_offsets + 1))
        duration_offsets = data_augmentation_offsets[2]
        self.duration_offsets = list(range(-duration_offsets, duration_offsets + 1))
        self.bar_fill_ratio = bar_fill_ratio
        self.bar_masking_duration_ratio_range = bar_masking_duration_ratio_range
        self.ac_random_ratio_range = ac_random_ratio_range
        self.seq2seq = seq2seq

        _paths = [
            Path(sample_["music"]["path"])
            for sample_ in self._dataset.select(
                list(range(MAX_NUM_FILES_NUM_TOKENS_PER_NOTE))
            )
        ]
        self.average_num_tokens_per_note = get_average_num_tokens_per_note(
            tokenizer, _paths
        )

        # Infill tokens, set as attribute here to avoid to access to vocab dic
        self._infill_bar_token = "Infill_Bar"
        self._infill_bar_start_token = "FillBar_Start"
        self._infill_bar_end_token = "FillBar_End"
        self._infill_bar_token_id = tokenizer.vocab["Infill_Bar"]
        self._infill_bar_start_token_id = tokenizer.vocab["FillBar_Start"]
        self._infill_bar_end_token_id = tokenizer.vocab["FillBar_End"]

        # Token ids that should be masked from the "labels" entry so that the loss is
        # not computed other them. Doing so, the model will not be trained to predict
        # them.
        self._token_ids_no_loss = torch.concat(
            [
                torch.LongTensor(
                    [self._infill_bar_token_id, self._infill_bar_start_token_id]
                ),
                torch.LongTensor(
                    [
                        tokenizer.vocab[token]
                        for ac in tokenizer.attribute_controls
                        for token in ac.tokens
                    ]
                ),
            ]
        )

        self._ac_tracks, self._ac_bars = [], []
        for i, ac in enumerate(tokenizer.attribute_controls):
            if isinstance(ac, BarAttributeControl):
                self._ac_bars.append(i)
            else:
                self._ac_tracks.append(i)

        super().__init__(
            [],
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
        self._max_seq_len = self.max_seq_len - sum(
            [1 for t in [bos_token_id, eos_token_id] if t is not None]
        )
        self.decoder_key_name = decoder_key_name

    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """
        Return the ``idx`` elements of the dataset.

        If the dataset is pre-tokenized, the method will return the token ids.
        Otherwise, it will tokenize the ``idx``th file on the fly. If the file to is
        corrupted, the method will return a dictionary with ``None`` values.

        :param idx: idx of the file/sample.
        :return: the token ids, with optionally the associated label.
        """
        labels = None

        # The tokenization steps are outside the try bloc as if there are errors,
        # we might want to catch them to fix them instead of skipping the iteration.
        try:
            score = Score.from_midi(self._dataset[idx]["music"]["bytes"])
        except SCORE_LOADING_EXCEPTION:
            item = {self.sample_key_name: None}
            if self.func_to_get_labels is not None:
                item[self.labels_key_name] = labels
            return item

        tseq, decoder_input_ids = self._tokenize_score(score)
        # If not one_token_stream, we only take the first track/sequence
        token_ids = tseq.ids if self.tokenizer.one_token_stream else tseq[0].ids
        if self.func_to_get_labels is not None:
            # tokseq can be given as a list of TokSequence to get the labels
            labels = self.func_to_get_labels(score, tseq, self.files_paths[idx])
            if not isinstance(labels, LongTensor):
                labels = LongTensor([labels] if isinstance(labels, int) else labels)

        item = {self.sample_key_name: LongTensor(token_ids)}
        if self.func_to_get_labels is not None:
            item[self.labels_key_name] = labels

        if self.seq2seq:
            item[self.decoder_key_name] = LongTensor(decoder_input_ids.ids)
            item[self.labels_key_name] = item[self.decoder_key_name].clone()
        else:
            item[self.labels_key_name] = item[self.sample_key_name].clone()
        idx_tokens_to_discard = torch.isin(
            item[self.labels_key_name], self._token_ids_no_loss
        )
        item[self.labels_key_name][idx_tokens_to_discard] = -100
        return item

    def _tokenize_score(self, score: Score) -> tuple[TokSequence, TokSequence | None]:
        # Delete unused elements in order to compute the bar ticks that we will use,
        # otherwise if unused elements are at the end of the score they can give us
        # bar ticks exceeding the tokens
        score.markers = []
        score.key_signatures = []
        for track in score.tracks:
            track.controls = []
            track.lyrics = []
            if not self.tokenizer.config.use_sustain_pedals:
                track.pedals = []
            if not self.tokenizer.config.use_pitch_bends:
                track.pitch_bends = []

        # Select k tracks (shuffled)
        # This is done before splitting per note density in order to have the right
        # estimated token sequence lengths.
        num_tracks_to_keep = max(
            1, round(len(score.tracks) * uniform(*self.ratio_random_tracks_range))
        )
        score.tracks = sample(list(score.tracks), k=num_tracks_to_keep)

        # Remove time signatures and tempos occurring after the start of the last note
        max_note_time = 0
        for track in score.tracks:
            if len(track.notes) > 0:
                max_note_time = max(max_note_time, track.notes[-1].time)
        for ti in reversed(range(len(score.time_signatures))):
            if score.time_signatures[ti].time > max_note_time:
                del score.time_signatures[ti]
            else:
                break
        for ti in reversed(range(len(score.tempos))):
            if score.tempos[ti].time > max_note_time:
                del score.tempos[ti]
            else:
                break

        # Split the input score into smaller chunks
        score_chunks = split_score_per_note_density(
            score,
            self.max_seq_len,
            self.average_num_tokens_per_note,
            num_overlap_bars=0,
            min_seq_len=MIN_SEQ_LEN,
        )
        score_chunks = [deepcopy(chunk) for chunk in score_chunks]
        shuffle(score_chunks)

        # Select the chunk to proceed with, augment and preprocess it while making sure
        # it remains valid. After preprocess, a chunk might contain no more notes if
        # they were all outside the pitch range of the tokenizer. If it is not, another
        # chunk is selected until it is or that there are no remaining chunk to select.
        # It can happen that there are no chunk to select from the beginning if the
        # original file is too short and the only possible chunk makes less than the
        # minimum sequence length provided to split_score_per_note_density. In this
        # case (that should be rare), we return an empty sequence.
        score.tracks = []  # delete them to enter the loop
        while len(score.tracks) == 0 and len(score_chunks) > 0:
            # We select the first to be able to select another one in the case where the
            # chunk is not valid.
            score = score_chunks.pop(0)
            # Augment the Score with randomly selected offsets among possible ones and
            # preprocess it.
            score = self.augment_and_preprocess_score(score)
        if len(score.tracks) == 0:
            return TokSequence(
                ids=[
                    self.tokenizer.vocab["Track_Start"],
                    self.tokenizer.vocab["Track_End"],
                ],
                tokens=["Track_Start", "Track_End"],
            ), TokSequence(
                ids=[self._infill_bar_start_token_id, self._infill_bar_end_token_id],
                tokens=[
                    self._infill_bar_start_token,
                    self._infill_bar_end_token,
                ],
            )

        # Need to preprocess here to know the number of tracks, i.e. sequences
        # If there is only one track, we do bar infilling as new track gen is not
        # possible in seq2seq.
        bar_infilling = len(score.tracks) == 1 or random() < self.bar_fill_ratio
        track_infilling_idx = None
        bar_tick_start, bar_tick_end, bar_section_length = None, None, None
        if bar_infilling:
            track_infilling_idx = choice(list(range(len(score.tracks))))
            # ac_indexes contains random bar acs only for the section to infill
            bars_ticks = np.array(get_bars_ticks(score))
            bars_ticks = bars_ticks[  # remove bars ticks after end of track
                np.where(bars_ticks <= score.tracks[track_infilling_idx].notes[-1].time)
            ]
            bar_section_length = max(
                1,
                round(
                    len(bars_ticks) * uniform(*self.bar_masking_duration_ratio_range)
                ),
            )
            if bar_section_length == len(bars_ticks):
                # Special case where the portion to mask == the whole music
                bar_idx_start = 0
            else:
                bar_idx_start = choice(
                    list(range(len(bars_ticks) - bar_section_length))
                )
            bar_idx_end = bar_idx_start + bar_section_length
            bar_tick_start = bars_ticks[bar_idx_start]
            bar_tick_end = (
                bars_ticks[bar_idx_end] if bar_idx_end < len(bars_ticks) else None
            )

            # Set ac_indexes
            acs_idx = sample(
                population=self._ac_bars,
                k=round(
                    len(self._ac_bars) * uniform(*self.tracks_idx_random_ratio_range)
                ),
            )

            ac_indexes = {
                track_infilling_idx: {
                    ac_idx: sample(
                        population=list(
                            range(
                                bar_idx_start,
                                bar_tick_end + 1 if bar_tick_end else len(bars_ticks),
                            )
                        ),
                        k=round(
                            bar_section_length
                            * uniform(*self.tracks_idx_random_ratio_range)
                        ),
                    )
                    for ac_idx in acs_idx
                }
            }
        else:
            # ac_indexes only contains random track acs for the last track
            acs_idx = sample(
                population=self._ac_tracks,
                k=round(
                    len(self._ac_tracks) * uniform(*self.tracks_idx_random_ratio_range)
                ),
            )
            ac_indexes = {len(score.tracks) - 1: {ac_idx: True for ac_idx in acs_idx}}

        # Tokenize it
        sequences = self.tokenizer.encode(
            score,
            encode_ids=False,
            no_preprocess_score=True,
            attribute_controls_indexes=ac_indexes,
            concatenate_track_sequences=False,
        )

        # Remove track sequences so that it doesn't exceed max_seq_len
        # First remove track sequences that exceed the max_seq_len
        if bar_infilling:
            for i in reversed(range(len(sequences))):
                if len(sequences[i]) > self._max_seq_len:
                    if len(sequences) == 1 or i == track_infilling_idx:
                        # Ideally this shouldn't happen
                        # We could cut the length of the track_infilling_idx track but
                        # it cannot be over bars to infill...
                        return TokSequence(
                            ids=[
                                self.tokenizer.vocab["Track_Start"],
                                self.tokenizer.vocab["Track_End"],
                            ],
                            tokens=["Track_Start", "Track_End"],
                        ), TokSequence(
                            ids=[
                                self._infill_bar_start_token_id,
                                self._infill_bar_end_token_id,
                            ],
                            tokens=[
                                self._infill_bar_start_token,
                                self._infill_bar_end_token,
                            ],
                        )
                    del sequences[i]
                    if i < track_infilling_idx != 0:
                        track_infilling_idx -= 1
            # Then makes sure that the sum of all sequences doesn't exceed max_seq_len
            sequences_copy = deepcopy(sequences)
            while sum(len(seq) for seq in sequences_copy) > self._max_seq_len:
                population = list(range(len(sequences_copy)))
                if track_infilling_idx:
                    population.remove(track_infilling_idx)
                idx_to_del = choice(population)
                del sequences_copy[idx_to_del]
                if track_infilling_idx and track_infilling_idx > idx_to_del:
                    track_infilling_idx -= 1
            sequences = sequences_copy

        # If doing bar infilling we place extract the bars and create place the
        # right infilling tokens
        # Otherwise (track infilling), there is nothing to do here. If a user wants to
        # create a new track, we'll just have to add Track_Start and Program tokens at
        # the end of the sequence and generate from here.
        decoder_input_ids = None
        if bar_infilling:
            # Bar infilling
            # Extract token section of the bars to infill
            # We do not select tracks that end before bar_tick_start
            times = np.array(
                [event.time for event in sequences[track_infilling_idx].events]
            )
            if bar_tick_start == 0:
                # excluding Track_Start and attribute control token
                token_idx_start = 0
                while (
                    sequences[track_infilling_idx].events[token_idx_start].type_
                    != "Bar"
                ):
                    token_idx_start += 1
            else:
                token_idx_start = np.nonzero(times >= bar_tick_start)[0]
                token_idx_start = token_idx_start[0]
            if bar_tick_end is None:
                # excluding Track_End and attribute control token
                token_idx_end = -2
            else:
                token_idx_end = np.nonzero(times >= bar_tick_end)[0]
                token_idx_end = token_idx_end[0] if len(token_idx_end) > 0 else -2

            # Extract the tokens of the section to infill and add BarFill start/end
            seq_infill = sequences[track_infilling_idx][token_idx_start:token_idx_end]
            seq_infill.ids.insert(0, self._infill_bar_start_token_id)
            seq_infill.tokens.insert(0, self._infill_bar_start_token)
            seq_infill.ids.append(self._infill_bar_end_token_id)
            seq_infill.tokens.append(self._infill_bar_end_token)
            # seq2seq --> decoder infill / concat seq before after
            if self.seq2seq:
                decoder_input_ids = seq_infill
            else:
                # Adding it at the end of the list that will be flattened
                sequences.append(seq_infill)

            # Add BarInfill tokens + update sequences
            seq_before = sequences[track_infilling_idx][:token_idx_start]
            for _ in range(bar_section_length):
                seq_before.ids.append(self._infill_bar_token_id)
                seq_before.tokens.append(self._infill_bar_token)
            seq_after = sequences[track_infilling_idx][token_idx_end:]
            sequences[track_infilling_idx] = seq_before + seq_after
        else:
            # There are always at least two sequences
            decoder_input_ids = sequences.pop(choice(list(range(len(sequences)))))

        # TODO External labels: (non-)expressive, loops, genres to add to the seq

        # Preprocessing token ids: reduce sequence length, add BOS/EOS tokens
        tokseq = concat_tokseq(sequences)
        self.tokenizer.encode_token_ids(tokseq)
        tokseq.ids = self._preprocess_token_ids(
            tokseq.ids,
            self.max_seq_len,
            self.bos_token_id,
            self.eos_token_id,
            enforce_eos_token_if_seq_len_exceed_lim=False,
        )
        return tokseq, decoder_input_ids

    def augment_and_preprocess_score(self, score: Score) -> Score:
        """
        Augment a ``symusic.Score`` and preprocess it with the tokenizer.

        :param score: score to augment and preprocess.
        :return: the augmented and preprocessed track.
        """
        pitch_offsets = _filter_offset_tuples_to_score(
            self.pitch_offsets.copy(),
            score,
            restrict_on_program_tessitura=True,
        )
        if len(pitch_offsets) > 0:
            score = augment_score(
                score,
                choice(pitch_offsets),
                choice(self.velocity_offsets),
                choice(self.duration_offsets),
            )
        return self.tokenizer.preprocess_score(score)

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        :return: number of elements in the dataset.
        """
        return len(self._dataset)

    def __str__(self) -> str:  # noqa:D105
        return f"{len(self)} samples."
