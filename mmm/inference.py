"""Inference method for the MMM model."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from miditok import MMM, TokSequence
from symusic import Score
from torch import LongTensor
from transformers import LogitsProcessor, LogitsProcessorList


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from .logits_processor import StopLogitsProcessor
    from .config import InferenceConfig


def generate(
    model: object,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    score_or_path: Score | Path | str,
    logits_processor : StopLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
) -> Score:
    """
    Use the model to generate new music content.

    The method allows to infill specific bars or generate new tracks.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param inference_config: InferenceConfig
    :param score_or_path: ``symusic.Score`` or path of the music file to infill.
    :param logits_processor: ``transformers.LogitsProcessor`` used to stop generation when the right number
        of bars is generated.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    score = (
        Score(score_or_path) if not isinstance(score_or_path, Score) else score_or_path
    )

    # Infill bars
    if inference_config.infilling:
        score = generate_infilling(
            model, tokenizer, inference_config, score, logits_processor, generate_kwargs
        )

    # Generate new tracks
    if inference_config.autoregressive:
        for track in inference_config.new_tracks:
            score = generate_new_track(model, tokenizer, track, score, generate_kwargs)

    return score


def generate_new_track(
    model: object,
    tokenizer: MMM,
    track: tuple[int, list[str]],
    score: Score,
    generate_kwargs: Mapping | None = None,
) -> Score:
    """
    Generate a new track of a given Score.

    The new track will be added to the score.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track: tuple containing the program of the track and a list of Track
        Attribute Controls.
    :param score: symusic.Score
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    if not generate_kwargs:
        generate_kwargs = {}

    # In this case, the prompt is a toksequence containing all the tracks
    input_seq = tokenizer.encode(score)

    # Add <TRACK_START> and <PROGRAM> tokens
    input_seq.ids.append(tokenizer.vocab["Track_Start"])
    input_seq.tokens.append("Track_Start")
    input_seq.ids.append(tokenizer.vocab[f"Program_{track[0]}"])
    input_seq.tokens.append(f"Program_{track[0]}")

    # Add attribute control tokens
    for control in track[1]:
        input_seq.ids.append(tokenizer.vocab[control])
        input_seq.tokens.append(control)

    output_ids = model.generate(LongTensor([input_seq.ids]), **generate_kwargs)
    output_seq = TokSequence(ids=output_ids[0].tolist(), are_ids_encoded=True)

    # Remove attribute controls from the sequence
    output_seq = (
        output_seq[: len(input_seq)] + output_seq[len(input_seq) + len(track[1]) :]
    )

    # Decode BPE ids before getting the associated tokens
    tokenizer.decode_token_ids(output_seq)
    output_seq.tokens = tokenizer._ids_to_tokens(output_seq.ids)

    # It is expected to have a <TRACK_END> token at the end of the sequence.
    if output_seq.tokens[-1] != "Track_End":
        warnings.warn(
            "Track generation failed: the model failed to predict a <TRACK_END> token",
            stacklevel=2,
        )
        output_seq.ids.append(tokenizer.vocab["Track_End"])
        output_seq.tokens.append("Track_End")

    return tokenizer._tokens_to_score(output_seq)


def generate_infilling(
    model: object,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    score: Score,
    logits_processor : StopLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
) -> Score:
    """
    Generate a new portion of a ``symusic.Score``.

    The portion to infill will be generated with the model and added to the score
    inplace for the selected tracks. Notes originally present in the portion to
    infill will be removed.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param score: ``symusic.Score`` to generate a new track from.
    :param inference_config: InferenceConfig
    :param logits_processor: ``transformers.LogitsProcessor`` used to stop generation when the right number
        of bars is generated.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    if not generate_kwargs:
        generate_kwargs = {}

    tracks_to_infill = inference_config.bars_to_generate.keys()
    input_tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    for track_to_infill in tracks_to_infill:
        infill_bars(
            model,
            tokenizer,
            track_to_infill,
            inference_config,
            input_tokens,
            logits_processor,
            generate_kwargs,
        )

    # Here we use the base tokenizer because output_tokens is a list of TokSequences
    return tokenizer.base_tokenizer._tokens_to_score(input_tokens)


def infill_bars(
    model: object,
    tokenizer: MMM,
    track_idx: int,
    inference_config: InferenceConfig,
    tokens: list[TokSequence],
    logits_processor : StopLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
) -> None:
    """
    Infill bars for the ''track_idx'' track.

    The tokens are replaced inplace.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param inference_config: contains information about which tracks and bars to
        generate.
    :param tokens: TokSequence of the track to be infilled
    :param logits_processor: ``transformers.LogitsProcessor`` used to stop generation when the right number
        of bars is generated.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    """
    if not generate_kwargs:
        generate_kwargs = {}

    # For each set of bars to infill in the track, we generate new content
    # (We may have, in the same track, non-adjacent sequences of bars. For
    # each sequence, we do a generation step).
    for subset_bars_to_infill in inference_config.bars_to_generate[track_idx]:
        input_seq = _adapt_prompt_for_bar_infilling(
            tokenizer, track_idx, tokens, subset_bars_to_infill
        )

        logits_processor.n_bars_to_infill = subset_bars_to_infill[1] - subset_bars_to_infill[0]
        logit_processor_list = LogitsProcessorList()
        logit_processor_list.append(logits_processor)

        output_ids = model.generate(LongTensor([input_seq.ids]), logits_processor=logit_processor_list ,**generate_kwargs)[
            0
        ].numpy()

        fill_start_idx = np.where(output_ids == tokenizer.vocab["FillBar_Start"])[0][0]
        track_end_idx = np.where(output_ids == tokenizer.vocab["Track_End"])[0][0]

        #fill_end_idx = np.where(output_ids == tokenizer.vocab["FillBar_End"])[0][0]
        #infill_bar_idxs = np.where(output_ids == tokenizer.vocab["Infill_Bar"])[0]

        # For debugging purposes
        generated_tokens = TokSequence(are_ids_encoded=True)
        generated_tokens.ids = output_ids[fill_start_idx:].tolist()
        tokenizer.decode_token_ids(generated_tokens)
        generated_tokens.tokens = tokenizer._ids_to_tokens(generated_tokens.ids)

        # Open the file in write mode ('w') and write tokens
        with open("output.txt", "w") as file:
            # Write each token on a new line
            file.write("\n".join(generated_tokens.tokens))

        replacing_tokens = TokSequence(are_ids_encoded=True)

        # subset_bars_to_infill[2] is the list of attribute controls
        replacing_tokens.ids = np.append(
            output_ids[: infill_bar_idxs[0]],
            output_ids[
                fill_start_idx + 1 + len(subset_bars_to_infill[2]) : fill_end_idx
            ],
        ).tolist()
        replacing_tokens.ids = np.append(
            replacing_tokens.ids,
            output_ids[infill_bar_idxs[-1] + 1 : track_end_idx + 1],
        ).tolist()

        # Decode BPE ids before getting the associated tokens
        tokenizer.decode_token_ids(replacing_tokens)
        replacing_tokens.tokens = tokenizer._ids_to_tokens(replacing_tokens.ids)

        # The model is assumed to generate Bar_None at the right position
        tokens[track_idx] = replacing_tokens


def _adapt_prompt_for_bar_infilling(
    tokenizer: MMM,
    track_idx: int,
    tokens: list[TokSequence],
    subset_bars_to_infill: tuple[int, int, list[str]],
) -> TokSequence:
    """
    Construct the prompt for bar infilling.

    Constructs the prompt to be used as model's input. The sequence should have the
    "BAR_FILL" format:
    ``<TRACK_START>...<TRACK_END>...<TRACKS_START>...<INFILL_BAR>...<INFILL_BAR>...
    <TRACK_END>...<TRACK_START>...<TRACK_END><START_FILL>``
    We have as many <FILL_IN> tokens as the number of bars we want to infill.

    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param tokens: TokSequence of the track to be infilled
    :param subset_bars_to_infill: contains the indexes of the first and last bar to
        infill, plus a list of attribute controls
    """
    output_toksequence: TokSequence = TokSequence()

    start_bar_idx = subset_bars_to_infill[0]
    end_bar_idx = subset_bars_to_infill[1]

    bars_ticks = tokens[track_idx]._ticks_bars
    bar_tick_start = bars_ticks[start_bar_idx]
    bar_tick_end = bars_ticks[end_bar_idx]

    times = np.array([event.time for event in tokens[track_idx].events])

    token_idx_start = np.nonzero(times >= bar_tick_start)[0]
    token_idx_start = token_idx_start[0]

    token_idx_end = np.nonzero(times >= bar_tick_end)[0]
    token_idx_end = token_idx_end[0]

    # Decode BPE tokens: this is necessary to put <INFILL_BAR> tokens
    # at the right place
    tokenizer.decode_token_ids(tokens[track_idx])

    seq_before = tokens[track_idx][:token_idx_start]
    for _ in range(end_bar_idx - start_bar_idx):
        seq_before.ids.append(tokenizer.vocab["Infill_Bar"])
        seq_before.tokens.append("Infill_Bar")
    seq_after = tokens[track_idx][token_idx_end:]
    output_toksequence += seq_before + seq_after

    # Encode into BPE tokens
    tokenizer.encode_token_ids(output_toksequence)

    output_toksequence.ids.append(tokenizer.vocab["FillBar_Start"])
    output_toksequence.tokens.append("FillBar_Start")

    attribute_controls = subset_bars_to_infill[2]
    for control in attribute_controls:
        output_toksequence.ids.append(tokenizer.vocab[control])
        output_toksequence.tokens.append(control)

    return output_toksequence
