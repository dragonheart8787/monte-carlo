"""
模組 5：接收與判決模組

介面契約（Protocol）：
- Detector:  input: received[n] complex -> output: symbol_indices[n] int
- Demapper:  input: symbol_idx int -> output: bits[k] int
- Compensator: input: received[n] complex, optional channel_state -> output: compensated[n] complex
- Receiver: 編排 compensator -> detector -> demapper
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

from .modulation import Modulator


# ---------------------------------------------------------------------------
# 介面契約 (Protocol)
# ---------------------------------------------------------------------------

@runtime_checkable
class DetectorProtocol(Protocol):
    """
    Detector 介面契約
    - input:  received symbols, shape (n,), dtype complex
    - output: symbol indices, shape (n,), dtype int, range [0, M-1]
    """
    def detect(self, received: np.ndarray) -> np.ndarray:
        """received: (n,) complex -> symbol_indices: (n,) int"""
        ...


@runtime_checkable
class DemapperProtocol(Protocol):
    """
    Demapper 介面契約
    - input:  symbol_idx, int in [0, M-1]
    - output: bits, shape (k,) int, k = bits_per_symbol
    """
    def symbol_to_bits(self, symbol_idx: int) -> np.ndarray:
        """symbol_idx: int -> bits: (k,) int"""
        ...

    @property
    def bits_per_symbol(self) -> int:
        """回傳每符號位元數"""
        ...


@runtime_checkable
class CompensatorProtocol(Protocol):
    """
    Compensator 介面契約
    - input:  received (n,) complex, channel_state 見 ChannelState 契約
    - output: compensated (n,) complex
    channel_state 欄位：phase_offset_rad, freq_offset_norm, fading_coefficients,
    perfect_csi_available, impairment_metadata
    """
    def compensate(
        self,
        received: np.ndarray,
        channel_state: Optional[dict] = None,
    ) -> np.ndarray:
        """received: (n,) complex -> compensated: (n,) complex"""
        ...


# ---------------------------------------------------------------------------
# Demapper 實作
# ---------------------------------------------------------------------------

class Demapper(ABC):
    """符號索引到位元的映射"""

    @abstractmethod
    def symbol_to_bits(self, symbol_idx: int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def bits_per_symbol(self) -> int:
        pass


class DemapperBPSK(Demapper):
    def symbol_to_bits(self, symbol_idx: int) -> np.ndarray:
        return np.array([1 - symbol_idx], dtype=int)

    @property
    def bits_per_symbol(self) -> int:
        return 1


class DemapperQPSK(Demapper):
    def symbol_to_bits(self, symbol_idx: int) -> np.ndarray:
        return np.array([(symbol_idx >> 1) & 1, symbol_idx & 1], dtype=int)

    @property
    def bits_per_symbol(self) -> int:
        return 2


class Demapper8PSK(Demapper):
    def symbol_to_bits(self, symbol_idx: int) -> np.ndarray:
        return np.array(
            [(symbol_idx >> 2) & 1, (symbol_idx >> 1) & 1, symbol_idx & 1],
            dtype=int,
        )

    @property
    def bits_per_symbol(self) -> int:
        return 3


class Demapper16QAM(Demapper):
    def symbol_to_bits(self, symbol_idx: int) -> np.ndarray:
        return np.array(
            [
                (symbol_idx >> 3) & 1,
                (symbol_idx >> 2) & 1,
                (symbol_idx >> 1) & 1,
                symbol_idx & 1,
            ],
            dtype=int,
        )

    @property
    def bits_per_symbol(self) -> int:
        return 4


# ---------------------------------------------------------------------------
# Detector 實作
# ---------------------------------------------------------------------------

class Detector(ABC):
    """符號檢測器"""

    @abstractmethod
    def detect(self, received: np.ndarray) -> np.ndarray:
        """received: (n,) complex -> symbol_indices: (n,) int"""
        pass


class MinimumDistanceDetector(Detector):
    def __init__(self, constellation: np.ndarray):
        self.constellation = np.asarray(constellation)

    def detect(self, received: np.ndarray) -> np.ndarray:
        dist = np.abs(received[:, np.newaxis] - self.constellation)
        return np.argmin(dist, axis=1)


# ---------------------------------------------------------------------------
# Compensator 實作
# ---------------------------------------------------------------------------

class ChannelCompensator(ABC):
    """通道補償"""

    @abstractmethod
    def compensate(
        self,
        received: np.ndarray,
        channel_state: Optional[dict] = None,
    ) -> np.ndarray:
        pass


class PhaseCompensator(ChannelCompensator):
    def __init__(self, phase_rad: float):
        self.phase = phase_rad

    def compensate(
        self,
        received: np.ndarray,
        channel_state: Optional[dict] = None,
    ) -> np.ndarray:
        return received * np.exp(-1j * self.phase)


class NoCompensation(ChannelCompensator):
    def compensate(
        self,
        received: np.ndarray,
        channel_state: Optional[dict] = None,
    ) -> np.ndarray:
        return received


class FadingEqualizer(ChannelCompensator):
    """
    MRC (Maximum Ratio Combining) equalizer for fading channels.
    Applies: r_mrc = conj(h) * received

    For BPSK/QPSK with MinimumDistanceDetector, detection reduces to
    sign(Re(r_mrc)), which correctly averages to the Rayleigh BER formula.
    ZF (dividing by |h|²) is avoided due to noise amplification in deep fades.
    Falls back to NoCompensation if h not available.
    """
    def compensate(
        self,
        received: np.ndarray,
        channel_state=None,
    ) -> np.ndarray:
        h = None
        if channel_state is not None:
            if hasattr(channel_state, "fading_coefficients"):
                h = channel_state.fading_coefficients
            elif isinstance(channel_state, dict):
                h = channel_state.get("fading_coefficients")

        if h is None:
            return received

        h_arr = np.asarray(h, dtype=complex)
        return received * np.conj(h_arr)


# ---------------------------------------------------------------------------
# Receiver 編排
# ---------------------------------------------------------------------------

def _get_demapper(modulation: str) -> Demapper:
    m = modulation.upper()
    if m == "BPSK":
        return DemapperBPSK()
    if m == "QPSK":
        return DemapperQPSK()
    if m == "8PSK":
        return Demapper8PSK()
    if m == "16QAM":
        return Demapper16QAM()
    raise ValueError(f"不支援的調變: {modulation}")


class Receiver:
    """編排 compensator -> detector -> demapper"""

    def __init__(
        self,
        modulation: str = "BPSK",
        decision_type: str = "hard",
        compensator: Optional[ChannelCompensator] = None,
    ):
        self.modulator = Modulator(modulation)
        self.decision_type = decision_type
        self.demapper: Demapper = _get_demapper(modulation)
        self.detector: Detector = MinimumDistanceDetector(
            self.modulator.get_constellation()
        )
        self.compensator: ChannelCompensator = compensator or NoCompensation()

    def detect(
        self,
        received: np.ndarray,
        channel_state: Optional[dict] = None,
    ) -> np.ndarray:
        """完整流程：compensate -> detect -> demap -> bits"""
        compensated = self.compensator.compensate(received, channel_state)
        symbol_indices = self.detector.detect(compensated)
        bits_list = [
            self.demapper.symbol_to_bits(int(idx))
            for idx in symbol_indices
        ]
        return np.concatenate(bits_list) if bits_list else np.array([], dtype=int)

    def demodulate(self, received: np.ndarray) -> np.ndarray:
        """向後相容"""
        return self.detect(received)
