from typing import Callable

import tiktoken


class BaseTokenizer:
    _required_special_tokens: list[str]
    _name_to_special_token: dict[str, str]
    _name_to_special_id: dict[str, int]

    def __init_subclass__(cls):
        for tok_name in cls._required_special_tokens:
            tok_getter = cls._special_token_getter(tok_name)
            tok_setter = cls._special_token_setter(tok_name)
            tok_idx_getter = cls._special_token_idx_getter(tok_name)
            tok_idx_setter = cls._special_token_idx_setter(tok_name)
            setattr(cls, tok_name, property(tok_getter, tok_setter))
            setattr(cls, f"{tok_name}_id", property(tok_idx_getter, tok_idx_setter))

    def __init__(
        self,
        base_encoding_name: str,
        model_max_len: int,
        **special_tokens: str,
    ) -> None:
        base_encoding = tiktoken.get_encoding(encoding_name=base_encoding_name)
        self._name_to_special_token = special_tokens
        self._name_to_special_id = self._get_name_to_special_ids(**special_tokens)
        self.special_tokens = {
            self._name_to_special_token[k]: self._name_to_special_id[k]
            for k in special_tokens
        }
        self.encoding = self._get_encoding_from_base(base_encoding, self.special_tokens)
        self.model_max_len = model_max_len

    @staticmethod
    def _special_token_getter(tok_name: str) -> Callable[["BaseTokenizer"], str]:
        def _get_special_token(self) -> str:
            try:
                return self._name_to_special_token[tok_name]
            except KeyError:
                raise AttributeError(f"Special token {tok_name} not set.")

        return _get_special_token

    @staticmethod
    def _special_token_setter(tok_name: str) -> Callable[["BaseTokenizer", str], None]:
        def _set_special_token(self, value: str) -> None:
            self._name_to_special_token[tok_name] = value

        return _set_special_token

    @staticmethod
    def _special_token_idx_getter(tok_name: str) -> Callable[["BaseTokenizer"], str]:
        def _get_special_token_id(self) -> str:
            try:
                return self._name_to_special_id[tok_name]
            except KeyError:
                raise AttributeError(f"Special token {tok_name} not set.")

        return _get_special_token_id

    @staticmethod
    def _special_token_idx_setter(
        tok_name: str,
    ) -> Callable[["BaseTokenizer", int], None]:
        def _set_special_token_id(self, value: int) -> None:
            self._name_to_special_id[tok_name] = value

        return _set_special_token_id

    @classmethod
    def _get_name_to_special_ids(cls, **special_tokens: str) -> dict[str, int]:
        if missing := set(cls._required_special_tokens) - set(special_tokens):
            raise ValueError(f"Missing required special tokens: {missing}")
        additional = set(special_tokens) - set(cls._required_special_tokens)
        name_to_special_id = dict[str, int]()
        for i, tok_name in enumerate(cls._required_special_tokens):
            name_to_special_id[tok_name] = i
        for i, tok_name in enumerate(
            additional, start=len(cls._required_special_tokens)
        ):
            name_to_special_id[tok_name] = i
        return name_to_special_id

    @classmethod
    def _get_encoding_from_base(
        cls, base_encoding: tiktoken.Encoding, special_tokens: dict[str, int]
    ) -> tiktoken.Encoding:
        num_special = len(special_tokens)
        mergable_ranks = {
            b: i + num_special for b, i in base_encoding._mergeable_ranks.items()
        }
        return tiktoken.Encoding(
            "tokenizer",
            pat_str=base_encoding._pat_str,
            mergeable_ranks=mergable_ranks,
            special_tokens=special_tokens,
            explicit_n_vocab=len(mergable_ranks) + num_special,
        )

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab


if __name__ == "__main__":
    tokenizer = BaseTokenizer(
        "r50k_base",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        model_max_len=512,
    )
    print("Derived Encoding: ")
    print(tokenizer.encoding.special_tokens_set)
    print(len(tokenizer.encoding._mergeable_ranks))
    i = 0
    for k, v in tokenizer.encoding._mergeable_ranks.items():
        if i == 5:
            break
        print(f"Mergeable rank: bpe {k!r}, index {v}")
        i += 1
    print(tokenizer.encoding.n_vocab)
    print(tokenizer.encoding.max_token_value + 1)
    # print(tokenizer.bos_token_id)
    # print(tokenizer.eos_token_id)
