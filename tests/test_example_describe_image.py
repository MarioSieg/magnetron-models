# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""The ordering the image example depends on, checked against a stub engine.

Loading a real snapshot needs weights and a tokenizer download, so what is pinned here is the part that
is easy to get wrong and silent when wrong: the placeholders go in front of the question, and the same
decoded images reach `gen_stream`. A prompt whose placeholders do not match the images it is generated
with produces a plausible answer about nothing.
"""

import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'examples'))
import describe_image  # noqa: E402


class _StubEngine:
    """Stands in for InferenceEngine, recording what the example hands it."""

    supports_images = True

    def __init__(self, tokens: int = 4) -> None:
        self.tokens = tokens
        self.config = SimpleNamespace(model='qwen3.5-4b', system='You are a helpful assistant.')
        self.model = SimpleNamespace(build_prompt=self._build_prompt)
        self.seen: dict[str, object] = {}

    def _build_prompt(self, system: str, messages: list[tuple[str, str]]) -> str:
        self.seen['system'] = system
        self.seen['messages'] = messages
        return f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{messages[0][1]}<|im_end|>\n'

    def load_images(self, paths: list[str]) -> list[str]:
        self.seen['paths'] = paths
        return [f'image:{p}' for p in paths]

    def vision_prefix(self, images: list[str]) -> str:
        return '<|vision_start|>' + '<|image_pad|>' * self.tokens + '<|vision_end|>'

    def gen_stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        self.seen['prompt'] = prompt
        self.seen['kwargs'] = kwargs
        yield 'a red '
        yield 'square'


def test_placeholders_precede_the_question() -> None:
    engine = _StubEngine()
    answer = describe_image.describe(engine, 'photo.jpg', 'What is this?', max_tokens=16)
    assert answer == 'a red square'
    user_turn = engine.seen['messages'][0][1]
    assert user_turn.index('<|vision_end|>') < user_turn.index('What is this?'), 'the image must come before the text'
    assert user_turn.count('<|image_pad|>') == engine.tokens


def test_the_decoded_images_are_passed_to_generation() -> None:
    engine = _StubEngine()
    describe_image.describe(engine, 'photo.jpg', 'What is this?', max_tokens=16)
    assert engine.seen['paths'] == ['photo.jpg']
    # Same objects load_images returned: re-decoding elsewhere would risk a different grid than the prompt assumed.
    assert engine.seen['kwargs']['images'] == ['image:photo.jpg']
    assert engine.seen['kwargs']['reset_cache'] is True
    assert engine.seen['kwargs']['max_tokens'] == 16


def test_a_text_only_model_is_refused_before_any_work() -> None:
    engine = _StubEngine()
    engine.supports_images = False
    with pytest.raises(SystemExit, match='text only'):
        describe_image.describe(engine, 'photo.jpg', 'What is this?', max_tokens=16)
    assert 'paths' not in engine.seen, 'the image must not be decoded for a model that cannot use it'
