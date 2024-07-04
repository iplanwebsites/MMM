"""Lists the Experiment baselines and training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import MistralConfig, MistralForCausalLM

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
