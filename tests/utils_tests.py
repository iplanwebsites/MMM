"""Utils for tests."""

from pathlib import Path

import torch
from miditok import MMM
from symusic import Score
from torch import LongTensor

HERE = Path(__file__).parent
MIDI_PATHS = list((HERE / "midis").glob("**/*.mid"))
MIDI_PATH = HERE / "midis" / "I Gotta Feeling.mid"


class DummyModel:
    """Dummy model used to test generation code."""

    def __init__(self, tokenizer: MMM) -> None:
        self.tokenizer = tokenizer
        self._bar_infilling_token_id = self.tokenizer["Infill_Bar"]
        score = Score(MIDI_PATH)
        # TODO extract the desired portion from the score tokenize it and save attribute
        self.token_ids = token_ids

    def generate(self, token_ids: LongTensor) -> LongTensor:
        """
        Generate method.

        :param token_ids: token ids.
        :return: supposedly generated token ids.
        """
        # If we have to infill bars
        if torch.any(token_ids == self._bar_infilling_token_id):
            generated_tokens = []
            token_ids = torch.cat(
                (
                    token_ids,
                    LongTensor(generated_tokens).reshape(1, len(generated_tokens)),
                ),
                dim=1,
            )

        # otherwise, we are generating a new track
        else:
            generated_tokens = []
            token_ids = torch.cat(
                (
                    token_ids,
                    LongTensor(generated_tokens).reshape(1, len(generated_tokens)),
                ),
                dim=1,
            )

        return token_ids
