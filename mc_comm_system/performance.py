"""
模組 6：效能評估模組

系統 KPI 中心：
- BER (Bit Error Rate)
- SER (Symbol Error Rate)
- PER (Packet Error Rate)
- 信賴區間 (Confidence Interval)
- 收斂相關指標
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


class PerformanceEvaluator:
    """效能指標計算與統計分析"""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def compute_ber(
        self, tx_bits: np.ndarray, rx_bits: np.ndarray
    ) -> float:
        """計算位元錯誤率"""
        n = min(len(tx_bits), len(rx_bits))
        errors = np.sum(tx_bits[:n] != rx_bits[:n])
        return errors / n if n > 0 else 0.0

    def compute_ser(
        self, tx_symbols: np.ndarray, rx_symbols: np.ndarray
    ) -> float:
        """計算符號錯誤率"""
        n = min(len(tx_symbols), len(rx_symbols))
        errors = np.sum(tx_symbols[:n] != rx_symbols[:n])
        return errors / n if n > 0 else 0.0

    def compute_per(
        self,
        tx_bits: np.ndarray,
        rx_bits: np.ndarray,
        packet_starts: List[int],
        packet_length: int,
    ) -> float:
        """計算封包錯誤率（任一 bit 錯即該封包錯）"""
        if not packet_starts:
            return 0.0
        packet_errors = 0
        for start in packet_starts:
            end = start + packet_length
            if end <= len(tx_bits) and end <= len(rx_bits):
                if np.any(tx_bits[start:end] != rx_bits[start:end]):
                    packet_errors += 1
        return packet_errors / len(packet_starts)

    def confidence_interval(
        self, samples: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        計算樣本平均的信賴區間。
        回傳 (mean, lower, upper)
        """
        n = len(samples)
        if n < 2:
            return float(np.mean(samples)), float(np.mean(samples)), float(np.mean(samples))
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        se = std / np.sqrt(n)
        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin = z * se
        return mean, mean - margin, mean + margin

    def standard_error(self, p: float, n: int) -> float:
        """BER/SER 的標準誤差: sqrt(p(1-p)/n)"""
        if n <= 0:
            return 0.0
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.sqrt(p * (1 - p) / n)

    def aggregate_results(
        self,
        ber_samples: List[float],
        ser_samples: Optional[List[float]] = None,
        per_samples: Optional[List[float]] = None,
    ) -> Dict:
        """
        彙總多次試驗的結果，含信賴區間。
        """
        result = {}
        ber_arr = np.array(ber_samples)
        result["ber_mean"] = float(np.mean(ber_arr))
        result["ber_std"] = float(np.std(ber_arr, ddof=1)) if len(ber_arr) > 1 else 0.0
        result["ber_ci_low"], result["ber_ci_high"] = self._ci_for_proportion(
            result["ber_mean"], len(ber_arr)
        )
        result["ber_se"] = self.standard_error(result["ber_mean"], len(ber_arr))

        if ser_samples:
            ser_arr = np.array(ser_samples)
            result["ser_mean"] = float(np.mean(ser_arr))
            result["ser_ci_low"], result["ser_ci_high"] = self._ci_for_proportion(
                result["ser_mean"], len(ser_arr)
            )

        if per_samples:
            per_arr = np.array(per_samples)
            result["per_mean"] = float(np.mean(per_arr))
            result["per_ci_low"], result["per_ci_high"] = self._ci_for_proportion(
                result["per_mean"], len(per_arr)
            )

        return result

    def _ci_for_proportion(
        self, p: float, n: int
    ) -> Tuple[float, float]:
        """二項比例的信賴區間（Wilson score）"""
        if n <= 0:
            return 0.0, 0.0
        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        low = max(0, center - margin)
        high = min(1, center + margin)
        return low, high
