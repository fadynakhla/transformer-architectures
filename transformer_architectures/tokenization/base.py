import abc

import tiktoken


class BaseTokenizer():
    def __init__(self, base_encoding_name: str, special_tokens: set[str], model_max_len: int) -> None:
        base_encoding = tiktoken.get_encoding(encoding_name=base_encoding_name)
        self.encoding = self._get_encoding_from_base(base_encoding, special_tokens)
        self.model_max_len = model_max_len

    @classmethod
    def _get_encoding_from_base(cls, base_encoding: tiktoken.Encoding, special_tokens: set[str]) -> tiktoken.Encoding:
        print(base_encoding.special_tokens_set)
        print(base_encoding.n_vocab)
        num_special = len(special_tokens)
        mergable_ranks = {b: i + num_special for b, i in base_encoding._mergeable_ranks.items()}
        special_tokens_map = {special: i for i, special in enumerate(special_tokens)}
        # n_vocab = base_encoding.n_vocab + num_special - len(base_encoding.special_tokens_set)
        return tiktoken.Encoding(
            "tokenizer",
            pat_str=base_encoding._pat_str,
            mergeable_ranks=mergable_ranks,
            special_tokens=special_tokens_map,
            # explicit_n_vocab=n_vocab
        )


if __name__=="__main__":
    tokenizer = BaseTokenizer("o200k_base", special_tokens={"<pad>"}, model_max_len=512)
    print(tokenizer.encoding._mergeable_ranks)
    print(tokenizer.encoding.special_tokens_set)
    print(tokenizer.encoding.n_vocab)
