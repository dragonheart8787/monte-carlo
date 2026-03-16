"""
理論模組測試

- BPSK AWGN 理論值單調下降
- 高 SNR 時 BER 明顯趨近更小
- Rayleigh BER 要比 AWGN 差
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_comm_system.theory import (
    ber_bpsk_awgn,
    ber_qpsk_awgn,
    ber_bpsk_rayleigh,
    ber_qpsk_rayleigh,
    get_theoretical_ber,
)


def test_bpsk_awgn_monotonic():
    """BPSK AWGN 理論 BER 隨 SNR 單調下降"""
    snr = np.array([0, 2, 4, 6, 8, 10, 12])
    ber = ber_bpsk_awgn(snr)
    for i in range(len(ber) - 1):
        assert ber[i] > ber[i + 1], f"BER 應單調下降: {ber[i]} > {ber[i+1]}"


def test_high_snr_ber_small():
    """高 SNR 時 BER 應很小"""
    ber = ber_bpsk_awgn(20)
    assert ber < 1e-8, f"20dB 時 BER 應 < 1e-8, 得到 {ber}"


def test_rayleigh_worse_than_awgn():
    """Rayleigh BER 應比 AWGN 差（同 SNR）"""
    snr = np.array([5, 10, 15])
    ber_awgn = ber_bpsk_awgn(snr)
    ber_rayleigh = ber_bpsk_rayleigh(snr)
    for i in range(len(snr)):
        assert ber_rayleigh[i] > ber_awgn[i], (
            f"Rayleigh 應比 AWGN 差 @ {snr[i]}dB"
        )


def test_get_theoretical_ber():
    """get_theoretical_ber 回傳正確類型"""
    r = get_theoretical_ber("BPSK", "AWGN", [0, 6, 12])
    assert r is not None
    assert len(r) == 3
    r_none = get_theoretical_ber("16QAM", "Rayleigh", [5])
    assert r_none is None or len(r_none) > 0


def test_qpsk_approx_bpsk_awgn():
    """QPSK AWGN 理論 BER 近似 BPSK"""
    snr = np.array([0, 6, 12])
    ber_b = ber_bpsk_awgn(snr)
    ber_q = ber_qpsk_awgn(snr)
    np.testing.assert_array_almost_equal(ber_b, ber_q)
