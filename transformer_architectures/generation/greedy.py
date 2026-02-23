from typing import Callable

import torch

DecodeFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def greedy_search(
    decode_fn: DecodeFn,
    encoder_output: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    max_length: int,
) -> torch.Tensor:
    """Generate sequences using greedy decoding.

    At each step, selects the token with the highest logit from the
    model output. Stops when all sequences in the batch have produced
    the eos_token_id or max_length is reached.

    Args:
        decode_fn: The model's decode method. Signature:
            (decoder_input, decoder_attention_mask,
             encoder_output, encoder_attention_mask) -> logits
            where logits has shape (batch, seq_len, vocab_size).
        encoder_output: Encoder hidden states, shape
            (batch, src_len, embed_dim).
        encoder_attention_mask: Encoder mask, shape (batch, src_len).
        bos_token_id: Token ID to seed the decoder sequence.
        eos_token_id: Token ID that signals end of generation.
        pad_token_id: Token ID used for padding after EOS.
        max_length: Maximum generated sequence length.

    Returns:
        Generated token IDs, shape (batch, generated_len).
    """
    batch_size = encoder_output.size(0)
    device = encoder_output.device

    decoder_input_ids = torch.full(
        (batch_size, 1), bos_token_id, dtype=torch.long, device=device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.uint8)

        logits = decode_fn(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_output,
            encoder_attention_mask,
        )

        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(finished, pad_token_id, next_token)

        decoder_input_ids = torch.cat(
            [decoder_input_ids, next_token.unsqueeze(-1)], dim=-1
        )

        finished = finished | (next_token == eos_token_id)

        if finished.all():
            break

    return decoder_input_ids
