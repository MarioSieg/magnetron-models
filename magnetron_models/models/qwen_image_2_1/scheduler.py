# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import math

from magnetron import Tensor, dtype
from .config import SchedulerConfig


class FlowMatchEulerScheduler:
    def __init__(self, cfg: SchedulerConfig) -> None:
        self.cfg = cfg
        self.sigmas: list[float] = []

    def shift_for(self, image_seq_len: int) -> float:
        c = self.cfg
        m = (c.max_shift - c.base_shift) / (c.max_image_seq_len - c.base_image_seq_len)
        b = c.base_shift - m * c.base_image_seq_len
        return image_seq_len * m + b

    def set_timesteps(self, num_inference_steps: int, image_seq_len: int) -> list[float]:
        if num_inference_steps < 1:
            raise ValueError(f'num_inference_steps must be at least 1, got {num_inference_steps}')
        n = num_inference_steps
        sigmas = [1.0 + (1.0 / n - 1.0) * i / max(n - 1, 1) for i in range(n)]
        e_mu = math.exp(self.shift_for(image_seq_len))
        sigmas = [e_mu / (e_mu + (1.0 / s - 1.0)) for s in sigmas]
        if self.cfg.shift_terminal and sigmas[-1] < 1.0:
            scale = (1.0 - sigmas[-1]) / (1.0 - self.cfg.shift_terminal)
            sigmas = [1.0 - (1.0 - s) / scale for s in sigmas]
        self.sigmas = [*sigmas, 0.0]
        return [s * self.cfg.num_train_timesteps for s in sigmas]

    @property
    def num_steps(self) -> int:
        return max(len(self.sigmas) - 1, 0)

    def step(self, model_output: Tensor, step_index: int, sample: Tensor) -> Tensor:
        if not 0 <= step_index < self.num_steps:
            raise IndexError(f'step {step_index} is outside the {self.num_steps} step schedule, call set_timesteps first')
        dt: float = self.sigmas[step_index + 1] - self.sigmas[step_index]
        prev = sample.cast(dtype.float32) + model_output.cast(dtype.float32) * dt
        return prev.cast(model_output.dtype)
