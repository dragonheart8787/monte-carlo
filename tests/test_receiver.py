"""
接收器測試

- zero-noise 下 AWGN + BPSK 必須接近零錯誤
- phase compensation 啟用時，在 phase offset 場景下 BER 應改善
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_comm_system.data_generator import DataGenerator
from mc_comm_system.modulation import Modulator
from mc_comm_system.channel import Channel
from mc_comm_system.receiver import Receiver, PhaseCompensator, NoCompensation


def test_zero_noise_bpsk_zero_error():
    """無雜訊時 BPSK 應零錯誤"""
    dg = DataGenerator(42)
    mod = Modulator("BPSK")
    ch = Channel("AWGN", random_seed=42)
    rx = Receiver("BPSK")

    bits = dg.generate_bits(1000)
    symbols = mod.modulate(bits)
    received = ch.transmit(symbols, 100.0)  # 極高 SNR ≈ 無雜訊
    rx_bits = rx.detect(received)

    errors = np.sum(bits != rx_bits)
    assert errors == 0, f"無雜訊時應零錯誤，得到 {errors} 個錯誤"


def test_phase_compensation_helps():
    """有相位偏移時，補償應改善 BER"""
    dg = DataGenerator(42)
    mod = Modulator("BPSK")
    phase_offset = np.pi / 4
    ch = Channel("AWGN", phase_offset_rad=phase_offset, random_seed=42)

    bits = dg.generate_bits(2000)
    symbols = mod.modulate(bits)
    received = ch.transmit(symbols, 8.0)

    rx_no_comp = Receiver("BPSK", compensator=NoCompensation())
    rx_comp = Receiver("BPSK", compensator=PhaseCompensator(phase_offset))

    ber_no = np.mean(bits != rx_no_comp.detect(received))
    ber_comp = np.mean(bits != rx_comp.detect(received))

    assert ber_comp < ber_no, f"補償後 BER ({ber_comp}) 應小於無補償 ({ber_no})"
