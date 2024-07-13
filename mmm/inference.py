from pathlib import Path

import numpy as np
import symusic
import torch
from utils.classes import InferenceConfig
from utils.constants import GENERATION_CONFIG_PARAMS
from miditok import MMM, TokSequence
from symusic import Score
from transformers import GenerationConfig, PreTrainedModel


def generate(
    model: PreTrainedModel,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    input_midi_path: str | Path,
    output_midi_path: str | Path
) -> None:
    """
    Generate new midi content for the bars and tracks specified in inference_config, and writes it to a new MIDI file.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param inference_config: InferenceConfig
    :param input_midi_path: path of the midi file to infill
    :param output_midi_path: path of the output midi file
    """
    score = symusic.Score(input_midi_path)

    if inference_config.infilling:
        score = generate_infilling(model, tokenizer, inference_config, score)
    if inference_config.autoregressive:
        for track in inference_config.new_tracks:
            new_tracks_score = generate_new_track(model, tokenizer, track, score)

    if inference_config.infilling:
        if inference_config.autoregressive:
            for i in range(len(inference_config.new_tracks)):
                score.tracks.append(new_tracks_score.tracks[len(score) + i])
        score.dump_midi(output_midi_path)
    else:
        new_tracks_score.dump_midi(output_midi_path)


def generate_new_track(model: PreTrainedModel, tokenizer: MMM, track: tuple[int, list[str]], score: Score) -> Score:
    """
    Generate a new track of a given Score.

    The new track will be generated with the model and

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track: tuple containing the program of the track and a list of Track Attribute Controls
    :param score: symusic.Score
    """
    # In this case, the prompt is the whole toksequence
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

    output_ids = model.generate(torch.tensor([input_seq.ids]), GenerationConfig(**GENERATION_CONFIG_PARAMS))

    output_seq = input_seq
    output_seq.ids = output_ids
    # tokenizer.complete_sequence(output_seq)
    output_seq.tokens += tokenizer._ids_to_tokens(output_ids[len(input_seq.tokens):])

    return tokenizer._tokens_to_score(output_seq)


def generate_infilling(
        model: PreTrainedModel,
        tokenizer: MMM,
        inference_config: InferenceConfig,
        score: Score
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
    """
    tracks_to_infill = inference_config.bars_to_generate.keys()
    input_tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    output_tokens = []
    for track_to_infill in tracks_to_infill:
        output_tokens.append(infill_bars(model, tokenizer, track_to_infill, inference_config,
                                         input_tokens))

    # Here we use the base tokenizer because output_tokens is a list of TokSequences
    return tokenizer.base_tokenizer._tokens_to_score(output_tokens)


def infill_bars(
        model: PreTrainedModel,
        tokenizer: MMM,
        track_idx: int,
        inference_config: InferenceConfig,
        tokens: TokSequence
) -> TokSequence:
    """
    Infill bars for the ''track_idx'' track.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param inference_config: contains information about which tracks and bars to generate
    :param tokens: TokSequence of the track to be infilled

    :return: Infilled TokSequence
    """

    input_seq = generate_infill_prompt(tokenizer, track_idx, inference_config, tokens)

    output_ids = model.generate(torch.tensor([input_seq.ids]), GenerationConfig(**GENERATION_CONFIG_PARAMS))
    output_ids = np.array(output_ids)

    # Move the generated content to replace the <FILL_IN> tokens.
    fill_start_idxs = np.where(output_ids == tokenizer.vocab["FillBar_Start"])[0]
    fill_end_idxs = np.where(output_ids == tokenizer.vocab["FillBar_End"])[0]
    infill_bar_idxs = np.where(output_ids == tokenizer.vocab["Infill_Bar"])[0]

    # TODO: Handle the case when the model predicts a number of <FILL_START> tokens different from <FILL_END>
    # different from <INFILL_BAR> tokens
    if len(infill_bar_idxs) != len(fill_start_idxs) != len(fill_end_idxs):
        msg = ""
        raise ValueError(msg)

    replacing_tokens = TokSequence()
    for i in range(len(infill_bar_idxs)):
        replacing_tokens.ids.append(tokenizer.vocab["Bar_None"])
        replacing_tokens.tokens.append("Bar_None")
        replacing_tokens.ids += output_ids[fill_start_idxs[i]:fill_end_idxs[i]]
        replacing_tokens.tokens += tokenizer._ids_to_tokens(output_ids[fill_start_idxs[i]:fill_end_idxs[i]].tolist())

    return tokens[:infill_bar_idxs[0]] + replacing_tokens + tokens[infill_bar_idxs[-1]:fill_start_idxs[0]]


def generate_infill_prompt(tokenizer: MMM, track_idx: int, inference_config: InferenceConfig,
                           tokens: TokSequence) -> TokSequence:
    """
    Constructs the prompt to be used as model's input. The sequence should have the "BAR_FILL" format:
    <TRACK_START>...<TRACK_END>...<TRACKS_START>...<FILL_IN>...<FILL_IN>...
    <TRACK_END>...<TRACK_START>...<TRACK_END><START_FILL>
    We have as many <FILL_IN> tokens as the number of bars we want to infill.

    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param inference_config: contains information about which tracks and bars to generate and attribute controls
    :param tokens: TokSequence of the track to be infilled
    """
    output_toksequence: TokSequence = TokSequence()
    for context_track_idx in inference_config.context_tracks:
        # If the track is the one to infill
        if context_track_idx == track_idx:
            start_bar_idx = inference_config.bars_to_generate[track_idx][0]
            end_bar_idx = inference_config.bars_to_generate[track_idx][1]

            bars_ticks = tokens[track_idx]._ticks_bars
            bar_tick_start = bars_ticks[start_bar_idx]
            bar_tick_end = bars_ticks[end_bar_idx]

            times = np.array([event.time for event in tokens[track_idx].events])

            token_idx_start = np.nonzero(times >= bar_tick_start)[0]
            token_idx_start = token_idx_start[0]

            token_idx_end = np.nonzero(times >= bar_tick_end)[0]
            token_idx_end = token_idx_end[0]

            seq_before = tokens[track_idx][:token_idx_start]
            for _ in range(end_bar_idx - start_bar_idx):
                seq_before.ids.append(tokenizer.vocab["Infill_Bar"])
                seq_before.tokens.append("Infill_Bar")
            seq_after = tokens[track_idx][token_idx_end:]
            output_toksequence += seq_before + seq_after

        output_toksequence += tokens[context_track_idx]

    output_toksequence.ids.append(tokenizer.vocab["FillBar_Start"])
    output_toksequence.tokens.append("FillBar_Start")

    attribute_controls = inference_config.bars_to_generate[track_idx][2]
    for control in attribute_controls:
        output_toksequence.ids.append(tokenizer.vocab[control])
        output_toksequence.tokens.append(control)

    return output_toksequence
