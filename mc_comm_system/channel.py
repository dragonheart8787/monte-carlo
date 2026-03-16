"""
模組 4：通道模組

內部拆層架構：
- FadingProcess: fading 增益
- AdditiveNoiseProcess: 加性雜訊
- ImpairmentProcess: 相位/頻率等擾動
- Channel: 編排 transmit = impairment(fading * symbols) + noise
"""

import numpy as np
from typing import Optional

from .config import SimulationConfig
from .channel_processes import (
    FadingProcess,
    AWGNFading,
    RayleighFading,
    RicianFading,
    AdditiveNoiseProcess,
    AWGNNoise,
    BurstAWGNNoise,
    ImpairmentProcess,
    NoImpairment,
    PhaseOffsetImpairment,
    FreqOffsetImpairment,
    CombinedImpairment,
)


class Channel:
    """通訊通道模型（編排 fading + impairment + noise）"""

    def __init__(
        self,
        channel_type: str = "AWGN",
        rician_k: float = 3.0,
        random_seed: Optional[int] = None,
        phase_offset_rad: float = 0.0,
        freq_offset_norm: float = 0.0,
        fading_mode: str = "fast",
        block_size: int = 100,
        burst_noise_prob: float = 0.0,
        burst_noise_ratio: float = 10.0,
    ):
        self.channel_type = channel_type.upper()
        self._random_seed = random_seed

        # Fading
        self._fading: FadingProcess = self._build_fading(
            self.channel_type, rician_k, fading_mode, block_size, random_seed
        )

        # Noise
        self._noise: AdditiveNoiseProcess = self._build_noise(
            burst_noise_prob, burst_noise_ratio, random_seed
        )

        # Impairment
        self._impairment: ImpairmentProcess = self._build_impairment(
            phase_offset_rad, freq_offset_norm
        )

    def _build_fading(
        self,
        ch_type: str,
        rician_k: float,
        fading_mode: str,
        block_size: int,
        seed: Optional[int],
    ) -> FadingProcess:
        if ch_type == "AWGN":
            return AWGNFading()
        if ch_type == "RAYLEIGH":
            return RayleighFading(fading_mode, block_size, seed)
        if ch_type == "RICIAN":
            return RicianFading(rician_k, fading_mode, block_size, seed)
        raise ValueError(f"不支援的通道類型: {ch_type}")

    def _build_noise(
        self,
        burst_prob: float,
        burst_ratio: float,
        seed: Optional[int],
    ) -> AdditiveNoiseProcess:
        if burst_prob > 0:
            return BurstAWGNNoise(burst_prob, burst_ratio, seed)
        return AWGNNoise(seed)

    def _build_impairment(
        self,
        phase_offset: float,
        freq_offset: float,
    ) -> ImpairmentProcess:
        parts = []
        if phase_offset != 0:
            parts.append(PhaseOffsetImpairment(phase_offset))
        if freq_offset != 0:
            parts.append(FreqOffsetImpairment(freq_offset))
        return CombinedImpairment(parts) if parts else NoImpairment()

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "Channel":
        """從 SimulationConfig 建立 Channel"""
        return cls(
            channel_type=config.channel,
            rician_k=config.rician_k,
            random_seed=config.random_seed,
            phase_offset_rad=config.phase_offset_rad,
            freq_offset_norm=config.freq_offset_norm,
            fading_mode=config.fading_mode,
            block_size=config.block_size,
            burst_noise_prob=config.burst_noise_prob,
            burst_noise_ratio=config.burst_noise_ratio,
        )

    def transmit(self, symbols: np.ndarray, snr_db: float) -> np.ndarray:
        """
        傳輸流程：fading -> impairment -> add noise
        """
        n = len(symbols)
        sig_power = np.mean(np.abs(symbols) ** 2)
        symbols_norm = symbols / np.sqrt(sig_power) if sig_power > 0 else symbols

        # 1. Fading
        h = self._fading.generate(n)
        faded = h * symbols_norm

        # 2. Impairment
        impaired = self._impairment.apply(faded)

        # 3. Additive noise
        noise = self._noise.generate(n, snr_db)

        return impaired + noise
