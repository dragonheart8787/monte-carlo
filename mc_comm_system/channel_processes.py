"""
通道內部拆層

責任分離：
- FadingProcess: 產生複數 fading 增益 h[n]
- AdditiveNoiseProcess: 產生加性雜訊（含 burst）
- ImpairmentProcess: 相位/頻率等擾動
- Channel: 編排三者
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from .config import SimulationConfig


# ---------------------------------------------------------------------------
# FadingProcess
# Input:  n (符號數)
# Output: h[n] 複數增益，每符號或每 block
# ---------------------------------------------------------------------------

class FadingProcess(ABC):
    """Fading 增益產生器"""

    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """產生長度 n 的複數 fading 增益"""
        pass


class AWGNFading(FadingProcess):
    """無 fading（增益恆為 1）"""
    def generate(self, n: int) -> np.ndarray:
        return np.ones(n, dtype=complex)


class RayleighFading(FadingProcess):
    """Rayleigh fading"""

    def __init__(self, fading_mode: str = "fast", block_size: int = 100, random_seed: Optional[int] = None):
        self.fading_mode = fading_mode.lower()
        self.block_size = block_size
        self.rng = np.random.default_rng(random_seed)

    def generate(self, n: int) -> np.ndarray:
        if self.fading_mode == "block":
            num_blocks = (n + self.block_size - 1) // self.block_size
            gains = (
                self.rng.standard_normal(num_blocks)
                + 1j * self.rng.standard_normal(num_blocks)
            ) / np.sqrt(2)
            h = np.zeros(n, dtype=complex)
            for i in range(num_blocks):
                start = i * self.block_size
                end = min(start + self.block_size, n)
                h[start:end] = gains[i]
            return h
        return (
            self.rng.standard_normal(n)
            + 1j * self.rng.standard_normal(n)
        ) / np.sqrt(2)


class RicianFading(FadingProcess):
    """Rician fading"""

    def __init__(
        self,
        k: float = 3.0,
        fading_mode: str = "fast",
        block_size: int = 100,
        random_seed: Optional[int] = None,
    ):
        self.k = k
        self.fading_mode = fading_mode.lower()
        self.block_size = block_size
        self.rng = np.random.default_rng(random_seed)

    def generate(self, n: int) -> np.ndarray:
        los = np.sqrt(self.k / (self.k + 1))
        scatter = 1 / np.sqrt(2 * (self.k + 1))
        if self.fading_mode == "block":
            num_blocks = (n + self.block_size - 1) // self.block_size
            gains = los + scatter * (
                self.rng.standard_normal(num_blocks)
                + 1j * self.rng.standard_normal(num_blocks)
            )
            h = np.zeros(n, dtype=complex)
            for i in range(num_blocks):
                start = i * self.block_size
                end = min(start + self.block_size, n)
                h[start:end] = gains[i]
            return h
        return los + scatter * (
            self.rng.standard_normal(n)
            + 1j * self.rng.standard_normal(n)
        )


# ---------------------------------------------------------------------------
# AdditiveNoiseProcess
# Input:  n (符號數), snr_db, signal_power (可選，用於正規化)
# Output: noise[n] 複數雜訊
# ---------------------------------------------------------------------------

class AdditiveNoiseProcess(ABC):
    """加性雜訊產生器"""

    @abstractmethod
    def generate(self, n: int, snr_db: float) -> np.ndarray:
        """產生長度 n 的複數雜訊"""
        pass


class AWGNNoise(AdditiveNoiseProcess):
    """加性白高斯雜訊"""

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.default_rng(random_seed)

    def generate(self, n: int, snr_db: float) -> np.ndarray:
        snr_linear = 10 ** (snr_db / 10)
        noise_var = 1.0 / snr_linear
        std = np.sqrt(noise_var / 2)
        return std * (
            self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
        )


class BurstAWGNNoise(AdditiveNoiseProcess):
    """含突發雜訊的 AWGN"""

    def __init__(
        self,
        burst_prob: float = 0.0,
        burst_ratio: float = 10.0,
        random_seed: Optional[int] = None,
    ):
        self.burst_prob = burst_prob
        self.burst_ratio = burst_ratio
        self.rng = np.random.default_rng(random_seed)

    def generate(self, n: int, snr_db: float) -> np.ndarray:
        snr_linear = 10 ** (snr_db / 10)
        noise_var = 1.0 / snr_linear
        std_base = np.sqrt(noise_var / 2)
        burst_mask = self.rng.random(n) < self.burst_prob
        std = np.where(
            burst_mask,
            std_base * np.sqrt(self.burst_ratio),
            std_base,
        )
        return std * (
            self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
        )


# ---------------------------------------------------------------------------
# ImpairmentProcess
# Input:  symbols[n] 複數
# Output: impaired[n] 複數（相位/頻率等擾動後）
# ---------------------------------------------------------------------------

class ImpairmentProcess(ABC):
    """通道擾動（相位、頻率等）"""

    @abstractmethod
    def apply(self, symbols: np.ndarray) -> np.ndarray:
        """對符號施加擾動"""
        pass


class NoImpairment(ImpairmentProcess):
    """無擾動"""
    def apply(self, symbols: np.ndarray) -> np.ndarray:
        return symbols.copy()


class PhaseOffsetImpairment(ImpairmentProcess):
    """相位偏移"""
    def __init__(self, phase_rad: float):
        self.phase = phase_rad

    def apply(self, symbols: np.ndarray) -> np.ndarray:
        if self.phase == 0:
            return symbols.copy()
        return symbols * np.exp(1j * self.phase)


class FreqOffsetImpairment(ImpairmentProcess):
    """頻率偏移"""
    def __init__(self, freq_offset_norm: float):
        self.freq_offset = freq_offset_norm

    def apply(self, symbols: np.ndarray) -> np.ndarray:
        if self.freq_offset == 0:
            return symbols.copy()
        n = len(symbols)
        t = np.arange(n, dtype=float)
        return symbols * np.exp(2j * np.pi * self.freq_offset * t)


class CombinedImpairment(ImpairmentProcess):
    """
    組合多種擾動。

    Pipeline 順序（固定，不可隨意變更）：
    1. phase_offset
    2. frequency_offset
    3. amplitude_distortion（若未來擴充）
    4. other impairments

    不同順序數學上不等價，結果無法比較。
    新增 impairment 時須依此順序插入。
    """
    PIPELINE_ORDER = ["phase_offset", "frequency_offset", "amplitude", "other"]

    def __init__(self, impairments: list):
        self.impairments = impairments

    def apply(self, symbols: np.ndarray) -> np.ndarray:
        out = symbols.copy()
        for imp in self.impairments:
            out = imp.apply(out)
        return out
