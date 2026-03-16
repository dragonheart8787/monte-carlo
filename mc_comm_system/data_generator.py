"""
模組 2：資料產生層

負責產生輸入 bit stream 和實驗樣本。
- 隨機 bit 產生
- 特定 pattern 測試
- 連續位元流模式 / 封包模式
"""

import numpy as np
from typing import Optional, Tuple, List


class DataGenerator:
    """產生模擬所需的位元資料"""

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.default_rng(random_seed)

    def generate_bits(self, num_bits: int) -> np.ndarray:
        """產生隨機位元流 (0 或 1)"""
        return self.rng.integers(0, 2, size=num_bits)

    def generate_packets(
        self, num_packets: int, packet_length: int
    ) -> List[np.ndarray]:
        """封包模式：產生多個固定長度封包"""
        return [
            self.generate_bits(packet_length) for _ in range(num_packets)
        ]

    def generate_pattern(self, pattern: str, repeat: int = 1) -> np.ndarray:
        """產生特定 pattern（用於除錯或測試）"""
        bits = []
        for _ in range(repeat):
            for c in pattern:
                bits.append(int(c))
        return np.array(bits)

    def get_bits_for_simulation(
        self,
        total_bits: int,
        packet_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[List[int]]]:
        """
        取得模擬所需位元。
        若 packet_length 為 None，回傳連續位元流。
        否則回傳封包列表與每個封包的起始索引。
        """
        if packet_length is None:
            return self.generate_bits(total_bits), None

        num_packets = total_bits // packet_length
        packets = self.generate_packets(num_packets, packet_length)
        bits = np.concatenate(packets)
        packet_starts = [i * packet_length for i in range(num_packets)]
        return bits, packet_starts
