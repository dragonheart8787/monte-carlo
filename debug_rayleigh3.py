import numpy as np
from mc_comm_system.channel import Channel
from mc_comm_system.receiver import FadingEqualizer
from mc_comm_system.theory import ber_bpsk_rayleigh
from scipy import special

rng = np.random.default_rng(999)

# 模擬 100000 個 BPSK 符號，Rayleigh channel
N = 100000
bits = rng.integers(0, 2, N)
s = np.where(bits == 1, 1.0+0j, -1.0+0j)

snr_tests = [0, 2, 4, 6, 8, 10]
print(f"{'SNR':>6} | {'Sim BER':>12} | {'Theory (this)':>14} | {'Theory Proakis':>16} | {'Err%':>7}")
print("-" * 65)

for snr_db in snr_tests:
    ch = Channel('Rayleigh', random_seed=snr_db*7)
    received, state = ch.transmit(s, float(snr_db), return_channel_state=True)
    h = np.array(state.fading_coefficients, dtype=complex)

    # MRC
    mrc = received * np.conj(h)
    decisions = np.sign(np.real(mrc))
    bit_decisions = np.where(decisions > 0, 1, 0)
    ber = np.mean(bit_decisions != bits)

    # Theory via Proakis
    gamma = 10 ** (snr_db / 10)
    theory_proakis = 0.5 * (1 - np.sqrt(gamma / (1 + gamma)))
    theory_code = ber_bpsk_rayleigh(snr_db)

    print(f"{snr_db:>6} | {ber:>12.4e} | {theory_code:>14.4e} | {theory_proakis:>16.4e} | {abs(ber-theory_proakis)/theory_proakis*100:>7.1f}%")

print()
print("Note: With N=100000 symbols, statistical uncertainty is ~0.1/sqrt(N) ≈ 0.0003")
