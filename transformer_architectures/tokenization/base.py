from typing import Optional

import tiktoken


class BaseTokenizer:
    def __init__(
        self,
        base_encoding_name: str,
        model_max_len: int,
        pad_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        additional_special_tokens: Optional[set[str]] = None,
    ) -> None:
        base_encoding = tiktoken.get_encoding(encoding_name=base_encoding_name)
        special_tokens = self._get_special_tokens(
            pad_token, eos_token, additional_special_tokens
        )
        self.encoding = self._get_encoding_from_base(base_encoding, special_tokens)
        self.model_max_len = model_max_len
        self.pad_token = pad_token
        self.pad_token_id = special_tokens[pad_token] if pad_token else None
        self.eos_token = eos_token
        self.eos_token_id = special_tokens[eos_token] if eos_token else None

    @classmethod
    def _get_special_tokens(
        cls,
        pad_token: Optional[str],
        eos_token: Optional[str],
        additional_special_tokens: Optional[set[str]],
    ) -> dict[str, int]:
        special_tokens = dict[str, int]()
        if pad_token:
            special_tokens[pad_token] = 0
        if eos_token:
            special_tokens[eos_token] = 1 if pad_token else 0
        if additional_special_tokens:
            assert pad_token not in additional_special_tokens
            assert eos_token not in additional_special_tokens
            special_tokens.update(
                {
                    tok: i + len(special_tokens)
                    for i, tok in enumerate(additional_special_tokens)
                }
            )
        return special_tokens

    @classmethod
    def _get_encoding_from_base(
        cls, base_encoding: tiktoken.Encoding, special_tokens: dict[str, int]
    ) -> tiktoken.Encoding:
        print(base_encoding.special_tokens_set)
        print(base_encoding.n_vocab)
        num_special = len(special_tokens)
        mergable_ranks = {
            b: i + num_special for b, i in base_encoding._mergeable_ranks.items()
        }
        # n_vocab = base_encoding.n_vocab + num_special - len(base_encoding.special_tokens_set)
        return tiktoken.Encoding(
            "tokenizer",
            pat_str=base_encoding._pat_str,
            mergeable_ranks=mergable_ranks,
            special_tokens=special_tokens,
            # explicit_n_vocab=n_vocab
        )


if __name__ == "__main__":
    tokenizer = BaseTokenizer("o200k_base", pad_token="<pad>", model_max_len=512)
    print(tokenizer.encoding._mergeable_ranks)
    print(tokenizer.encoding.special_tokens_set)
    print(tokenizer.encoding.n_vocab)
