"""Lists the Experiment baselines and training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Callable, List, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import MistralConfig, MistralForCausalLM, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils.classes import InferenceConfig
from miditok import TokSequence
from miditok.attribute_controls import BarAttributeControl, AttributeControl

import numpy as np

from utils.constants import (
    GENERATION_CONFIG_PARAMS
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from miditok import MusicTokenizer
    from symusic import Score


class MMM(MistralForCausalLM):
    """
    MMM model.

    During training, the model needs to know the ids of the attribute control and bar
    infilling tokens in order to discard the logits at these positions from the loss
    computation so that the model do not learn to predict them.

    During inference, a user might want to generate new tracks or infill specific
    portions of bars. As such, the model needs to know the ids of the bar infilling and
    program tokens.
    """

    def __init__(
        self, config: MistralConfig, tokenizer: MusicTokenizer | None = None
    ) -> None:
        super().__init__(config)
        self.generation_config = GenerationConfig(**GENERATION_CONFIG_PARAMS)
        if tokenizer:
            # Register attribute controls tokens
            for ac in tokenizer.attribute_controls:
                for token in ac.tokens:
                    token_str = token.replace(".", "§") if "." in token else token
                    self.register_buffer(
                        token_str, torch.LongTensor([tokenizer.vocab[token]])
                    )
            for program in tokenizer.config.programs:
                self.register_buffer(
                    f"Program_{program}",
                    torch.LongTensor([tokenizer.vocab[f"Program_{program}"]]),
                )
            for infill_token in ["Infill_Bar", "FillBar_Start", "FillBar_End"]:
                self.register_buffer(
                    infill_token,
                    torch.LongTensor([tokenizer.vocab[infill_token]]),
                )

        self._token_ids_no_loss = torch.concat(
            [
                self.Infill_Bar,
                self.FillBar_Start,
                self.FillBar_End,
                *[
                    token_id
                    for token, token_id in self.named_buffers()
                    if token.startswith("AC")
                ],
            ]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        r"""Override the forward method to discard special tokens from loss."""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Overridden here: discard logits from special tokens (AC, InfillStart)
            idx_tokens_to_discard = (input_ids in self._token_ids_no_loss) - 1
            logits_filtered = logits[~idx_tokens_to_discard].contiguous()
            labels_filtered = labels[~idx_tokens_to_discard].contiguous()

            # Shift so that tokens < n predict n
            shift_logits = logits_filtered[..., :-1, :].contiguous()
            shift_labels = labels_filtered[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss, *output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        inference_config: InferenceConfig = None,
        score: Score = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        if inference_config.autoregressive:
            for track in inference_config.new_tracks:
                self.generate_new_track(self, track, score)
        else:
            self.generate_infilling(score, inference_config)

    def generate_new_track(self, track: tuple[int, list[AttributeControl]], score: Score):
        """
               Generate a new track of a given Score.

               The new track will be generated with the model and

               :param track: tuple containing the program of the track and a list of Track Attribute Controls
               :param score: symusic.Score
               """
        #In this case, the prompt is the whole toksequence
        input_seq = self.tokenizer.encode(score)
        #TODO: Add <TRACK_START> <PROGRAM> and attribute controls tokens,

        #TODO: Should add the stopping condition for generation
        super().generate(torch.tensor(input_seq.ids), self.generation_config)

    def generate_infilling(
            self, score: Score, inference_config: InferenceConfig
    ) -> None:
        """
        Generate a new portion of a ``symusic.Score``.

        The portion to infill will be generated with the model and added to the score
        inplace for the selected tracks. Notes originally present in the portion to
        infill will be removed.
        TODO might need to provide a few additional arguments like a GenerationConfig

        :param score: ``symusic.Score`` to generate a new track from.
        :param inference_config: InferenceConfig
        """
        tracks_to_infill = inference_config.bars_to_generate.keys()
        for track_to_infill in tracks_to_infill:

            updated_score = self.infill_track(track_to_infill, inference_config, score)


        #TODO:After all these steps the score should have all the generated content, convert this back to midi

    def infill_track(
            self, track_idx: int, inference_config: InferenceConfig, score: Score
    ) -> Score:
        """
        Inner step function called by higher-level generate fuctions

        The step is used to extract the correct token sequence form the previous
        generation's result (or initial sequence if first step)

        :param track_idx: index of the track to infill
        :param score: symusic.Score of the MIDI input file

        :return: StepResult of current step
        """

        input_seq = self.generate_infill_prompt(track_idx, inference_config, score)

        #TODO: generate the output from the input sequence. I think it's needed to override the generation method as we
        # want to generate multiple tokens until we fill the whole bars

        n_bars = inference_config.bars_to_generate[track_idx][1] - inference_config.bars_to_generate[track_idx][0]
        output = self._generate_infill_output(input_seq, n_bars)

        #TODO: Get toksequence from output

        #TODO: Update score with the new infilled track
        result = ...

        return result

    def generate_infill_prompt(self, track_idx: int, inference_config: InferenceConfig, score: Score) -> TokSequence:
        """
        Constructs the prompt to be used as model input
        """

        #TODO:Need to add a reference to the tokenizer (?)
        tokens = self.tokenizer.encode(score, concatenate_track_sequences = False)

        output_toksequence: TokSequence
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
                for _ in range(start_bar_idx - end_bar_idx):
                    seq_before.ids.append(self._infill_bar_token_id)
                    seq_before.tokens.append(self._infill_bar_token)
                seq_after = tokens[track_idx][token_idx_end:]
                output_toksequence += seq_before + seq_after


            output_toksequence += tokens[context_track_idx]

        #TODO:Add <START_FILL> token
        #TODO:Add attribute controls tokens
        attribute_controls = inference_config.bars_to_generate[track_idx][2]

        return output_toksequence

    def _generate_infill_output(self, input_seq: TokSequence, n_bars: int):
        """
        Function that given a BAR_FILL representation of the input sequence, generates the content of the bars to
        infill. The sequence should look like <TRACK_START>...<TRACK_END>...<TRACKS_START>...<FILL_IN>...<FILL_IN>...
        <TRACK_END>...<TRACK_START>...<TRACK_END><START_FILL>. Where we have as many <FILL_IN> tokens as the number of
        bars we want to infill

        As output, we get the same tokensequence but with the generated content <START_FILL>...<END_FILL>,
        <START_FILL>...<END_FILL> n times as the number of bars

        :param input_seq: the input tokensequence to infill
        :param n_bars: number of bars to infill
        """
        bars_generated = 0
        for bi in range(n_bars):
            logits = super().generate(torch.tensor(input_seq.ids), self.generation_config)

