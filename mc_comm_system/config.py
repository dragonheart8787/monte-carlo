"""
模組 1：參數設定層

定義整個模擬實驗的條件，包括：
- 調變方式 (BPSK, QPSK, 8-PSK, 16-QAM)
- 通道類型 (AWGN, Rayleigh, Rician)
- SNR 範圍、樣本數、判決方式
- 編碼選項、隨機種子
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class SimulationConfig:
    """模擬實驗的完整參數配置"""

    # 調變參數
    modulation: str = "BPSK"  # BPSK, QPSK, 8PSK, 16QAM

    # 通道參數
    channel: str = "AWGN"  # AWGN, Rayleigh, Rician
    rician_k: float = 3.0  # Rician K 因子 (LOS 功率 / 散射功率)
    fading_mode: str = "fast"  # fast=每符號, block=每區塊
    block_size: int = 100  # block fading 時每區塊符號數

    # 擾動參數（第一優先）
    phase_offset_rad: float = 0.0  # 相位偏移 (rad)
    freq_offset_norm: float = 0.0  # 正規化頻率偏移 (Δf·T_sym)

    # 擾動參數（第二優先）
    burst_noise_prob: float = 0.0  # 突發雜訊發生機率
    burst_noise_ratio: float = 10.0  # 突發時雜訊放大倍數

    # SNR 範圍 (dB)
    snr_db_range: Tuple[float, float] = (0.0, 20.0)
    snr_db_step: float = 2.0

    # 樣本與試驗
    bits_per_simulation: int = 10000
    num_trials: int = 100
    packet_length: Optional[int] = None  # None = 連續位元流模式

    # 判決方式
    decision_type: str = "hard"  # hard, soft

    # 編碼
    coding: Optional[str] = None  # None, hamming, repetition

    # 隨機種子
    random_seed: Optional[int] = None

    # 蒙地卡羅引擎選項
    mc_level: int = 1  # 1=標準, 2=收斂監控, 3=變異數縮減, 4=自適應
    confidence_level: float = 0.95
    convergence_threshold: float = 1e-4  # 收斂停止門檻
    max_samples_per_snr: int = 1_000_000  # 單一 SNR 最大樣本數

    # 變異數縮減
    use_antithetic: bool = False
    use_control_variate: bool = False

    # 實驗等級標籤（供 summary / report 分類）
    experiment_type: str = "exploratory"  # validation, comparison, stress, exploratory
    experiment_rigor: str = "baseline"  # baseline, high, research
    theory_available: Optional[bool] = None  # None=自動推斷

    def get_snr_points(self) -> List[float]:
        """產生 SNR 掃描點"""
        low, high = self.snr_db_range
        points = []
        x = low
        while x <= high:
            points.append(x)
            x += self.snr_db_step
        return points

    def validate(self) -> None:
        """驗證參數合理性（委派給 validators）"""
        from .validators import validate_config
        validate_config(self)
