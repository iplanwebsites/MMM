"""Lists the Experiment baselines and training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import CrossEntropyLoss
from transformers import MistralConfig, MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

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

    def generate_new_track(self, score: Score, program: int) -> None:
        """
        Generate a new track of a given Score.

        The new track will be generated with the model and
        TODO might need to provide a few additional arguments like a GenerationConfig

        :param score: ``symusic.Score`` to generate a new track from.
        :param program: program of the track to generate. Give ``-1`` for drums.
        """
        # TODO implement: tokenize --> inject Infilling --> generate --> detok...

    def generate_infilling(
        self, score: Score, bar_idx: tuple[int, int], track_idx: Sequence[int]
    ) -> None:
        """
        Generate a new portion of a ``symusic.Score``.

        The portion to infill will be generated with the model and added to the score
        inplace for the selected tracks. Notes originally present in the portion to
        infill will be removed.
        TODO might need to provide a few additional arguments like a GenerationConfig

        :param score: ``symusic.Score`` to generate a new track from.
        :param bar_idx: tuple of bar numbers of the portion of the score to infill.
        :param track_idx: indexes of the tracks to infill.
        """
        # TODO implement: tokenize --> inject Infilling --> generate --> detok...
