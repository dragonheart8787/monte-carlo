"""
直接數值模擬，不依賴 Channel class
驗證 Rayleigh + MRC 的理論 BER
"""
import numpy as np
from scipy import special

rng = np.random.default_rng(0)
N = 200000

# BPSK: s = +1 or -1
bits = rng.integers(0, 2, N)
s = np.where(bits == 1, 1.0, -1.0).astype(complex)

print("=== 直接數值模擬 Rayleigh + MRC ===")
print(f"{'SNR':>5} | {'Direct BER':>12} | {'Theory':>12} | {'Err':>7}")
print("-" * 45)

for snr_db in [0, 2, 4, 6, 8, 10]:
    snr_lin = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1.0 / (2 * snr_lin))

    # Rayleigh h: E[|h|^2] = 1
    h = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    noise = noise_std * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

    # Transmit
    received = h * s + noise

    # MRC
    mrc = np.conj(h) * received
    re_mrc = np.real(mrc)

    # Decisions
    bit_dec = np.where(re_mrc > 0, 1, 0)
    ber = np.mean(bit_dec != bits)

    # Theory
    gamma = snr_lin
    theory = 0.5 * (1 - np.sqrt(gamma / (1 + gamma)))

    print(f"{snr_db:>5} | {ber:>12.4e} | {theory:>12.4e} | {abs(ber-theory)/theory*100:>6.1f}%")

print()
print("=== 驗證 Channel class 是否一致 ===")
import sys
sys.path.insert(0, '.')
from mc_comm_system.channel import Channel

for snr_db in [0, 2, 4, 6]:
    rng2 = np.random.default_rng(0)
    bits2 = rng2.integers(0, 2, 50000)
    s2 = np.where(bits2 == 1, 1.0+0j, -1.0+0j)

    ch = Channel('Rayleigh', random_seed=42)
    received2, state = ch.transmit(s2, float(snr_db), return_channel_state=True)
    h2 = np.array(state.fading_coefficients, dtype=complex)
    mrc2 = np.conj(h2) * received2
    bit_dec2 = np.where(np.real(mrc2) > 0, 1, 0)
    ber2 = np.mean(bit_dec2 != bits2)

    gamma = 10**(snr_db/10)
    theory = 0.5 * (1 - np.sqrt(gamma / (1 + gamma)))
    print(f"  SNR={snr_db:2d}dB | Channel BER={ber2:.4e} | Theory={theory:.4e}")
