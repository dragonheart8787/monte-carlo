"""
模組 3：調變模組

將 bit 轉成可送進通道的符號。
- BPSK, QPSK, 8-PSK, 16-QAM
- Symbol mapping
- Constellation 視覺化支援
"""

import numpy as np
from typing import Tuple, Dict, Any


class Modulator:
    """數位調變器"""

    # 各調變的 bits per symbol
    BPS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4}

    def __init__(self, modulation: str = "BPSK"):
        self.modulation = modulation.upper()
        self.bps = self.BPS_PER_SYMBOL.get(self.modulation, 1)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """將位元序列調變為複數符號"""
        if self.modulation == "BPSK":
            return self._bpsk(bits)
        elif self.modulation == "QPSK":
            return self._qpsk(bits)
        elif self.modulation == "8PSK":
            return self._8psk(bits)
        elif self.modulation == "16QAM":
            return self._16qam(bits)
        else:
            raise ValueError(f"不支援的調變方式: {self.modulation}")

    def _bpsk(self, bits: np.ndarray) -> np.ndarray:
        """BPSK: 0 -> +1, 1 -> -1 (實數)"""
        return 1 - 2 * bits

    def _qpsk(self, bits: np.ndarray) -> np.ndarray:
        """QPSK: Gray mapping"""
        n = len(bits) // 2
        symbols = np.zeros(n, dtype=complex)
        for i in range(n):
            b0, b1 = bits[2 * i], bits[2 * i + 1]
            re = 1 - 2 * b0
            im = 1 - 2 * b1
            symbols[i] = (re + 1j * im) / np.sqrt(2)
        return symbols

    def _8psk(self, bits: np.ndarray) -> np.ndarray:
        """8-PSK: Gray mapping"""
        n = len(bits) // 3
        symbols = np.zeros(n, dtype=complex)
        for i in range(n):
            idx = bits[3 * i] * 4 + bits[3 * i + 1] * 2 + bits[3 * i + 2]
            phase = np.pi / 4 + idx * np.pi / 4
            symbols[i] = np.exp(1j * phase)
        return symbols

    def _16qam(self, bits: np.ndarray) -> np.ndarray:
        """16-QAM: Gray mapping, 正規化平均功率為 1"""
        n = len(bits) // 4
        symbols = np.zeros(n, dtype=complex)
        scale = 1 / np.sqrt(10)  # 正規化
        for i in range(n):
            b = bits[4 * i : 4 * i + 4]
            re = (2 * (b[0] ^ b[1]) - 1) * (2 + (2 * b[1] - 1))
            im = (2 * (b[2] ^ b[3]) - 1) * (2 + (2 * b[3] - 1))
            symbols[i] = scale * (re + 1j * im)
        return symbols

    def get_constellation(self) -> np.ndarray:
        """取得星座圖點位（用於視覺化）"""
        if self.modulation == "BPSK":
            return np.array([-1.0, 1.0])
        elif self.modulation == "QPSK":
            return np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        elif self.modulation == "8PSK":
            return np.exp(1j * (np.pi / 4 + np.arange(8) * np.pi / 4))
        elif self.modulation == "16QAM":
            scale = 1 / np.sqrt(10)
            points = []
            for re in [-3, -1, 1, 3]:
                for im in [-3, -1, 1, 3]:
                    points.append(scale * (re + 1j * im))
            return np.array(points)
        return np.array([])

    def get_info(self) -> Dict[str, Any]:
        """取得調變資訊（功率效率、頻譜效率等）"""
        return {
            "modulation": self.modulation,
            "bits_per_symbol": self.bps,
            "constellation": self.get_constellation(),
        }
