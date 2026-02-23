from typing import Callable

import torch

DecodeFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def beam_search(
    decode_fn: DecodeFn,
    encoder_output: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    max_length: int,
    num_beams: int,
) -> torch.Tensor:
    """Generate sequences using beam search.

    Maintains num_beams hypotheses per batch element and selects the
    highest-scoring completed sequence for each.

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
        pad_token_id: Token ID for padding completed beams.
        max_length: Maximum generated sequence length.
        num_beams: Number of beams to maintain per batch element.

    Returns:
        Best sequences, shape (batch, generated_len). Returns the
        highest-scoring beam for each batch element.
    """
    batch_size = encoder_output.size(0)
    device = encoder_output.device

    encoder_output_expanded = _expand_for_beams(encoder_output, num_beams)
    encoder_mask_expanded = _expand_for_beams(encoder_attention_mask, num_beams)

    decoder_input_ids = torch.full(
        (batch_size * num_beams, 1),
        bos_token_id,
        dtype=torch.long,
        device=device,
    )

    beam_scores = torch.full((batch_size, num_beams), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0

    beam_finished = torch.zeros(batch_size, num_beams, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.uint8)

        logits = decode_fn(
            decoder_input_ids,
            decoder_attention_mask,
            encoder_output_expanded,
            encoder_mask_expanded,
        )

        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        vocab_size = log_probs.size(-1)

        log_probs = log_probs.view(batch_size, num_beams, vocab_size)

        if beam_finished.any():
            finished_mask = beam_finished.unsqueeze(-1).expand_as(log_probs)
            log_probs = log_probs.masked_fill(finished_mask, float("-inf"))
            log_probs[:, :, pad_token_id] = torch.where(
                beam_finished, 0.0, log_probs[:, :, pad_token_id]
            )

        next_scores = beam_scores.unsqueeze(-1) + log_probs

        next_scores_flat = next_scores.view(batch_size, -1)

        top_scores, top_indices = torch.topk(next_scores_flat, num_beams, dim=-1)

        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        batch_offsets = (
            torch.arange(batch_size, device=device).unsqueeze(-1) * num_beams
        )
        flat_beam_indices = (batch_offsets + beam_indices).view(-1)

        decoder_input_ids = decoder_input_ids[flat_beam_indices]
        decoder_input_ids = torch.cat(
            [decoder_input_ids, token_indices.view(-1, 1)], dim=-1
        )

        beam_scores = top_scores

        beam_finished = beam_finished.gather(1, beam_indices)
        beam_finished = beam_finished | (token_indices == eos_token_id)

        if beam_finished.all():
            break

    best_beam_idx = beam_scores.argmax(dim=-1)
    batch_offsets = torch.arange(batch_size, device=device) * num_beams
    best_flat_indices = batch_offsets + best_beam_idx
    best_sequences = decoder_input_ids[best_flat_indices]

    return best_sequences


def _expand_for_beams(tensor: torch.Tensor, num_beams: int) -> torch.Tensor:
    """Repeat each batch element num_beams times along dim 0.

    Args:
        tensor: Input of shape (batch, ...).
        num_beams: Number of times to repeat each element.

    Returns:
        Expanded tensor of shape (batch * num_beams, ...).
    """
    shape = tensor.shape
    expanded = tensor.unsqueeze(1).expand(shape[0], num_beams, *shape[1:])
    return expanded.reshape(shape[0] * num_beams, *shape[1:])
