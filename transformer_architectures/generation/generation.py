from typing import Literal

import torch

from transformer_architectures.generation import beam_search as beam_search_mod
from transformer_architectures.generation import greedy as greedy_mod

SearchAlgorithm = Literal["greedy", "beam"]


class GenerationMixin:
    """Mixin that adds a generate() method to encoder-decoder models.

    The host class must provide:
        - encode(encoder_input, encoder_attention_mask) -> torch.Tensor
        - decode(decoder_input, decoder_attention_mask,
                 encoder_output, encoder_attention_mask) -> torch.Tensor
        - vocab_size: int
    """

    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        max_length: int = 128,
        search: SearchAlgorithm = "greedy",
        num_beams: int = 4,
    ) -> torch.Tensor:
        """Generate token sequences from encoder input.

        Args:
            encoder_input_ids: Source token IDs, shape (batch, src_len).
            encoder_attention_mask: Source mask, shape (batch, src_len).
                1 for real tokens, 0 for padding.
            bos_token_id: Beginning-of-sequence token ID used to seed
                the decoder.
            eos_token_id: End-of-sequence token ID. Generation stops
                when every sequence in the batch has produced this token.
            pad_token_id: Padding token ID used to pad completed
                sequences.
            max_length: Maximum number of tokens to generate (including
                the initial BOS token).
            search: Decoding algorithm. One of "greedy" or "beam".
            num_beams: Number of beams for beam search. Ignored when
                search is "greedy".

        Returns:
            Generated token IDs, shape (batch, generated_len). Each
            sequence starts with bos_token_id and ends with eos_token_id
            (or is truncated at max_length).
        """
        encoder_output = self.encode(encoder_input_ids, encoder_attention_mask)  # type: ignore[attr-defined]

        match search:
            case "greedy":
                return greedy_mod.greedy_search(
                    decode_fn=self.decode,  # type: ignore[attr-defined]
                    encoder_output=encoder_output,
                    encoder_attention_mask=encoder_attention_mask,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    max_length=max_length,
                )
            case "beam":
                return beam_search_mod.beam_search(
                    decode_fn=self.decode,  # type: ignore[attr-defined]
                    encoder_output=encoder_output,
                    encoder_attention_mask=encoder_attention_mask,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    max_length=max_length,
                    num_beams=num_beams,
                )
            case _:
                raise ValueError(
                    f"Unknown search algorithm: {search!r}. "
                    f"Must be one of {SearchAlgorithm}."
                )
