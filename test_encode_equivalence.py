"""Quick script to confirm that prepending/appending special token strings
before encode_batch produces identical results to the current approach of
encoding first and wrapping with token IDs in Python."""

import multiprocessing

from src.ta.architectures.vanilla.tokenization import Tokenizer

tokenizer = Tokenizer("r50k_base", 512)

texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "",
    "Short",
    "A " * 200,  # longer input
]

# Current approach: encode then wrap in Python
current = tokenizer.encoding.encode_batch(
    texts, num_threads=multiprocessing.cpu_count()
)
current = [
    [tokenizer.bos_token_id] + seq + [tokenizer.eos_token_id] for seq in current
]

# Proposed approach: wrap strings then encode with allowed_special
wrapped = [f"{tokenizer.bos_token}{t}{tokenizer.eos_token}" for t in texts]
proposed = tokenizer.encoding.encode_batch(
    wrapped,
    num_threads=multiprocessing.cpu_count(),
    allowed_special={tokenizer.bos_token, tokenizer.eos_token},
)

for i, (c, p) in enumerate(zip(current, proposed)):
    assert c == p, f"Mismatch at index {i}:\n  current:  {c}\n  proposed: {p}"

print(f"All {len(texts)} inputs match.")
