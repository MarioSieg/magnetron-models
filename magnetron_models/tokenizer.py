# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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
    def __init__(self, tok: Tokenizer) -> None:
        self.tok = tok

    @classmethod
    def from_snapshot_metadata(cls, metadata: dict[str, Any]) -> HFTokenizer | None:
        tokenizer_json: str | None = metadata.get('tokenizer_json')
        return cls(Tokenizer.from_str(tokenizer_json)) if tokenizer_json else None

    @classmethod
    def from_repo(cls, repo_id: str) -> HFTokenizer:
        return cls(Tokenizer.from_file(download_or_ensure_resource(repo_id=repo_id, filename='tokenizer.json')))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, tok_id: list[int]) -> str:
        return self.tok.decode(tok_id)
