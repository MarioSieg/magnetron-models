# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from abc import ABC, abstractmethod

from magnetron_models.utils import download_or_ensure_resource
from tokenizers import Tokenizer


class TokenizerBase(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, tok_id: list[int]) -> str:
        raise NotImplementedError()


class HFTokenizer(TokenizerBase):
    def __init__(self, repo_id: str) -> None:
        tok_path = download_or_ensure_resource(repo_id=repo_id, filename='tokenizer.json')
        self.tok = Tokenizer.from_file(tok_path)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, tok_id: list[int]) -> str:
        return self.tok.decode(tok_id)
