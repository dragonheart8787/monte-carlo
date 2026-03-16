"""
理論基準模組

提供 closed-form 理論 BER/SER，用於：
- 模擬值 vs 理論值驗證
- relative / absolute error 分析
- 不同樣本數下逼近理論值的收斂速度評估
"""

import numpy as np
from scipy import special
from typing import Union, Optional


def ber_bpsk_awgn(snr_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """BPSK over AWGN 理論 BER = 0.5 * erfc(sqrt(SNR))"""
    snr_linear = 10 ** (np.asarray(snr_db) / 10)
    return 0.5 * special.erfc(np.sqrt(snr_linear))


def ber_qpsk_awgn(snr_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """QPSK over AWGN 理論 BER (Gray mapping 近似 = BPSK)"""
    return ber_bpsk_awgn(snr_db)


def ser_mpsk_awgn(
    snr_db: Union[float, np.ndarray], M: int
) -> Union[float, np.ndarray]:
    """
    M-PSK over AWGN 理論 SER (union bound 近似)
    P_s ≈ 2 * Q(sqrt(2*gamma*sin²(π/M)))
    """
    snr_linear = 10 ** (np.asarray(snr_db) / 10)
    return 2 * special.erfc(
        np.sqrt(2 * snr_linear * np.sin(np.pi / M) ** 2)
    ) / 2


def ber_mpsk_awgn(
    snr_db: Union[float, np.ndarray], M: int, k: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    M-PSK over AWGN 理論 BER (Gray mapping 近似)
    P_b ≈ P_s / k, k = log2(M)
    """
    k = k or int(np.log2(M))
    ser = ser_mpsk_awgn(snr_db, M)
    return ser / k


def ser_16qam_awgn(snr_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    16-QAM over AWGN 理論 SER
    P_s = 3 * Q(sqrt(4*gamma/5)) * (1 - 0.75*Q(sqrt(4*gamma/5)))
    正規化星座平均功率 = 1
    """
    snr_linear = 10 ** (np.asarray(snr_db) / 10)
    gamma = 4 * snr_linear / 10  # 每符號 SNR 對應的 scaling
    q_val = 0.5 * special.erfc(np.sqrt(gamma / 2))
    return 3 * q_val * (1 - 0.75 * q_val)


def ber_16qam_awgn(snr_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """16-QAM over AWGN 理論 BER (Gray mapping 近似)"""
    ser = ser_16qam_awgn(snr_db)
    return ser / 4


def ber_bpsk_rayleigh(snr_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    BPSK over Rayleigh fading 理論 BER (closed-form)
    P_b = 0.5 * (1 - sqrt(gamma_bar / (1 + gamma_bar)))
    gamma_bar = 平均 SNR
    """
    snr_linear = 10 ** (np.asarray(snr_db) / 10)
    return 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))


def ber_qpsk_rayleigh(snr_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    QPSK over Rayleigh fading 理論 BER
    近似：P_b ≈ 0.5 * (1 - sqrt(gamma_bar/(2+gamma_bar)))
    """
    snr_linear = 10 ** (np.asarray(snr_db) / 10)
    return 0.5 * (1 - np.sqrt(snr_linear / (2 + snr_linear)))


def ber_bpsk_rician(
    snr_db: Union[float, np.ndarray], K: float
) -> Union[float, np.ndarray]:
    """
    BPSK over Rician fading 理論 BER (近似)
    K = Rician K 因子
    """
    snr_linear = 10 ** (np.asarray(snr_db) / 10)
    # 簡化近似：高 K 時趨近 AWGN，低 K 時趨近 Rayleigh
    rayleigh_part = 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))
    awgn_part = ber_bpsk_awgn(snr_db)
    # 線性插值權重
    w = 1 / (1 + K)
    return w * rayleigh_part + (1 - w) * awgn_part


def get_theoretical_ber(
    modulation: str,
    channel: str,
    snr_db: Union[float, np.ndarray],
    rician_k: float = 3.0,
) -> Optional[Union[float, np.ndarray]]:
    """
    依調變與通道取得理論 BER。
    若無對應公式則回傳 None。
    """
    mod = modulation.upper()
    ch = channel.upper()

    if ch == "AWGN":
        if mod == "BPSK":
            return ber_bpsk_awgn(snr_db)
        if mod == "QPSK":
            return ber_qpsk_awgn(snr_db)
        if mod == "8PSK":
            return ber_mpsk_awgn(snr_db, 8)
        if mod == "16QAM":
            return ber_16qam_awgn(snr_db)

    if ch == "RAYLEIGH":
        if mod == "BPSK":
            return ber_bpsk_rayleigh(snr_db)
        if mod == "QPSK":
            return ber_qpsk_rayleigh(snr_db)

    if ch == "RICIAN":
        if mod == "BPSK":
            return ber_bpsk_rician(snr_db, rician_k)

    return None


def get_theoretical_ser(
    modulation: str,
    channel: str,
    snr_db: Union[float, np.ndarray],
) -> Optional[Union[float, np.ndarray]]:
    """依調變與通道取得理論 SER。"""
    mod = modulation.upper()
    ch = channel.upper()

    if ch == "AWGN":
        if mod == "BPSK":
            return ber_bpsk_awgn(snr_db)  # SER = BER for BPSK
        if mod == "QPSK":
            return ser_mpsk_awgn(snr_db, 4)
        if mod == "8PSK":
            return ser_mpsk_awgn(snr_db, 8)
        if mod == "16QAM":
            return ser_16qam_awgn(snr_db)

    return None


def relative_error(simulated: float, theoretical: float) -> float:
    """相對誤差 |sim - theory| / theory，theory=0 時回傳 inf"""
    if theoretical == 0:
        return np.inf if simulated != 0 else 0.0
    return abs(simulated - theoretical) / abs(theoretical)


def absolute_error(simulated: float, theoretical: float) -> float:
    """絕對誤差 |sim - theory|"""
    return abs(simulated - theoretical)
