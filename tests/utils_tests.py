"""Utils for tests."""

from pathlib import Path

from torch import LongTensor

HERE = Path(__file__).parent
MIDI_PATHS = list((HERE / "midis").glob("**/*.mid"))
MIDI_PATH = HERE / "midis" / "I Gotta Feeling.mid"


class DummyModel:
    """Dummy model used to test generation code."""

    def __init__(self) -> None:
        self.t = 0

    def generate(self, token_ids: LongTensor) -> LongTensor:
        """
        Generate method.

        :param token_ids: token ids.
        :return: supposedly generated token ids.
        """
        # TODO analyze if the instruction is to infill (Infill token somewhere) or gen a
        #  new track (last token should be a `Program`).
        #  Load a dummy MIDI file (use the first track), reduce to n bars if infilling n
        #  bars, tokenize it and concatenate the tokens to the input_ids.
        return token_ids
