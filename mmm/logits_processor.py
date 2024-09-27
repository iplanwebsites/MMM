import numpy as np
import torch
from miditok import TokSequence
from transformers import LogitsProcessor

class StopLogitsProcessor(LogitsProcessor):
    """
    Custom ``transformers.LogitsProcessor`` implementation that allows stopping generation when
    enough content to infill bars is generated.

    :param bar_start_token_id: ID of the token indicating the start of a bar.
    :param n_bars_to_infill: number of bars to be infilled in this generation step.
    :param eos_token_id: ID of the EOS (end of sequence) token. If the number of bars reaches `max_bars`,
    the EOS token will be forced to stop generation.
    """

    n_bars_to_infill: int = 0  # This should change at every generation step as we may need to infill a different
    # number of bars at each step
    n_attribute_controls: int = 0 # Number of attribute controls to skip when decoding using BPE


    def __init__(self, bar_start_token_id: int, eos_token_id: int, tokenizer):
        self.bar_start_token_id = bar_start_token_id
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):

        generated_tokens = TokSequence(are_ids_encoded=True)

        fill_start_idx = np.where(input_ids[0].numpy() == self.tokenizer.vocab["FillBar_Start"])[0][0]

        n_bar_none = 0
        if fill_start_idx + self.n_attribute_controls + 1 < len(input_ids[0]):
            generated_tokens.ids = input_ids[0][fill_start_idx+3:].tolist()
            self.tokenizer.decode_token_ids(generated_tokens)

            n_bar_none = len(np.where(np.array(generated_tokens.ids) == self.tokenizer.vocab["Bar_None"])[0])

        # If we reach the self.n_bars_to_infill + 1 BarStart token sampled, we have generated enough content
        if n_bar_none >= self.n_bars_to_infill:
            scores[:, :] = -999999.0  # Penalize all tokens
            # But enforce the sampling of EOS token to stop generation
            scores[:, self.eos_token_id] = 999999.0


        return scores