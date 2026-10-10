# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from magnetron import Tensor, context
from tokenizers import Tokenizer

from magnetron_models.inference import AUTO_DVC, DTYPES, resolve_device
from magnetron_models.models.whisper import SAMPLE_RATE, LogMelSpectrogram, WhisperModel, chunks, load_audio
from magnetron_models.utils import console


def snapshot_path_for(repo: str, dtype_name: str) -> str:
    return f'{repo.split("/")[-1].lower()}-{DTYPES[dtype_name].short_name}.mag'


class Transcriber:
    def __init__(self, snapshot: str, device: str = AUTO_DVC, compile: bool = False) -> None:
        start = time.perf_counter()
        context.stop_grad_recorder()
        if compile:
            context.enable_compiler(True)
        context.set_default_device(resolve_device(device))
        console.print(f'Loading model from snapshot: {snapshot}', style='dim')
        self.model = WhisperModel.from_snapshot(snapshot)
        metadata: dict[str, Any] = self.model.snapshot_metadata
        self.tokenizer = Tokenizer.from_str(metadata['tokenizer_json'])
        self.mel = LogMelSpectrogram(self.model.cfg.num_mel_bins)
        gen: dict[str, Any] = metadata.get('generation_config') or {}
        hf: dict[str, Any] = metadata['hf_config']
        self.lang_to_id: dict[str, int] = gen.get('lang_to_id', {})
        self.task_to_id: dict[str, int] = gen.get('task_to_id', {})
        self.multilingual: bool = bool(gen.get('is_multilingual', self.lang_to_id))
        self.sot: int = gen.get('decoder_start_token_id', hf['decoder_start_token_id'])
        self.eos: int = gen.get('eos_token_id', hf['eos_token_id'])
        self.no_timestamps: int = gen.get('no_timestamps_token_id', self.tokenizer.token_to_id('<|notimestamps|>'))
        self.suppress = self._bias(gen.get('suppress_tokens', hf.get('suppress_tokens')) or [])
        self.begin_suppress = self._bias(gen.get('begin_suppress_tokens', hf.get('begin_suppress_tokens')) or [])
        console.print(f'Ready in {time.perf_counter() - start:.2f}s', style='dim')

    def _bias(self, token_ids: list[int]) -> Tensor:
        bias = [0.0] * self.model.cfg.vocab_size
        for tok in token_ids:
            bias[tok] = float('-inf')
        return Tensor(bias, dtype=context.get_default_dtype())

    def _prompt(self, language: str | None, task: str) -> list[int]:
        if not self.multilingual:
            return [self.sot, self.no_timestamps]
        return [self.sot, self.lang_to_id[language], self.task_to_id[task], self.no_timestamps]

    def _lang_token(self, language: str) -> str:
        tok = language if language.startswith('<|') else f'<|{language.lower()}|>'
        if tok not in self.lang_to_id:
            raise ValueError(f'Unknown language {language}, choose one of {", ".join(k[2:-2] for k in self.lang_to_id)}')
        return tok

    def transcribe(self, path: str, language: str | None = None, task: str = 'transcribe', max_tokens: int = 224) -> str:
        audio = load_audio(path)
        duration: float = audio.shape[0] / SAMPLE_RATE
        console.print(f'{path}: {duration:.1f}s of audio', style='dim')
        lang_tok: str | None = self._lang_token(language) if self.multilingual and language else None
        start = time.perf_counter()
        parts: list[str] = []
        for offset, chunk in chunks(audio):
            mel = self.mel(chunk).cast(context.get_default_dtype())
            encoder_out = self.model.encode(mel.reshape(1, *mel.shape))
            if self.multilingual and lang_tok is None:
                lang_ids = list(self.lang_to_id.values())
                detected = self.model.detect_language(encoder_out, self.sot, lang_ids)
                lang_tok = next(k for k, v in self.lang_to_id.items() if v == detected)
                console.print(f'Detected language: {lang_tok[2:-2]}', style='dim')
            console.print(f'[dim][{offset:6.1f}s][/dim] ', end='')
            tokens: list[int] = []
            emitted: str = ''
            for tok in self.model.generate(encoder_out, self._prompt(lang_tok, task), self.eos, self.suppress, self.begin_suppress, max_tokens):
                tokens.append(tok)
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                if '�' not in text:
                    console.print(text[len(emitted) :], style='bold white', end='')
                    emitted = text
            console.print()
            parts.append(self.tokenizer.decode(tokens, skip_special_tokens=True).strip())
        elapsed = time.perf_counter() - start
        console.print(f'Transcribed {duration:.1f}s in {elapsed:.2f}s ({duration / elapsed:.1f}x realtime)', style='dim')
        return ' '.join(p for p in parts if p)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Speech-to-text with OpenAI Whisper on Magnetron')
    parser.add_argument('audio', nargs='+', help='Audio files (.wav, .flac or .mp3)')
    parser.add_argument('--model', type=str, default='openai/whisper-base', help='HF repo to convert when no --snapshot is given')
    parser.add_argument('--snapshot', type=str, default=None, help='Local .mag snapshot from convert-whisper')
    parser.add_argument('--dtype', type=str, default='float32', choices=sorted(DTYPES), help='Data type used when converting')
    parser.add_argument('--device', type=str, default=AUTO_DVC, help='cpu, cuda, cuda:1 or auto')
    parser.add_argument('--language', type=str, default=None, help='Spoken language, e.g. en or de; detected when omitted')
    parser.add_argument('--task', type=str, default='transcribe', choices=['transcribe', 'translate'], help='translate outputs English')
    parser.add_argument('--max-tokens', type=int, default=224, help='Maximum tokens per 30s chunk')
    parser.add_argument('--out', type=str, default=None, help='Write the transcript to this text file')
    parser.add_argument('--compile', action='store_true', help='Fuse and compile CPU operators into C kernels (same as MAG_COMPILE=1)')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    snapshot: str | None = args.snapshot
    if snapshot is None:
        snapshot = snapshot_path_for(args.model, args.dtype)
        if not os.path.exists(snapshot):
            from dataset_conversion_pipelines.convert_whisper import convert

            convert(args.model, DTYPES[args.dtype], snapshot)
    transcriber = Transcriber(snapshot, args.device, args.compile)
    results: list[str] = []
    for path in args.audio:
        text = transcriber.transcribe(path, args.language, args.task, args.max_tokens)
        console.print(f'\n{text}\n', style='bold green')
        results.append(text)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results) + '\n')


if __name__ == '__main__':
    main()
