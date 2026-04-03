import numpy as np
import sys; sys.path.insert(0, '.')
from mc_comm_system.channel import Channel
from mc_comm_system.channel_processes import AWGNNoise

rng = np.random.default_rng(0)
N = 50000
bits = rng.integers(0, 2, N)
s = np.where(bits == 1, 1.0+0j, -1.0+0j)

snr_db = 2.0
snr_lin = 10**(snr_db/10)

# Check 1: noise power from AWGNNoise
noise_gen = AWGNNoise(random_seed=0)
noise = noise_gen.generate(N, snr_db)
print(f"Expected noise power: {1/snr_lin:.4f}")
print(f"Actual noise power: {np.mean(np.abs(noise)**2):.4f}")
print(f"Expected noise std/comp: {np.sqrt(1/(2*snr_lin)):.4f}")
print(f"Actual noise Re std: {np.std(np.real(noise)):.4f}")

# Check 2: AWGN channel BER (no fading)
print()
ch_awgn = Channel('AWGN', random_seed=42)
rec_awgn = ch_awgn.transmit(s, snr_db)
ber_awgn = np.mean((np.sign(np.real(rec_awgn)) * np.real(s)) < 0)
from scipy.special import erfc
theory_awgn = 0.5 * erfc(np.sqrt(snr_lin))
print(f"AWGN: sim_BER={ber_awgn:.4f}, theory={theory_awgn:.4f}")

# Check 3: Rayleigh channel, but check the state h vs channel h
print()
ch_ray = Channel('Rayleigh', random_seed=42)
rec_ray, st = ch_ray.transmit(s, snr_db, return_channel_state=True)
h = np.array(st.fading_coefficients, dtype=complex)

print(f"|h|^2 mean: {np.mean(np.abs(h)**2):.4f}")
print(f"received power: {np.mean(np.abs(rec_ray)**2):.4f}")
print(f"Expected (|h|^2*signal + noise) power: ~{np.mean(np.abs(h)**2) + 1/snr_lin:.4f}")

# Reconstruct: what is the noise in received?
noise_ray = rec_ray - h * s
print(f"Reconstructed noise power: {np.mean(np.abs(noise_ray)**2):.4f}")
print(f"Expected noise power: {1/snr_lin:.4f}")

# BER without equalization
ber_raw = np.mean((np.sign(np.real(rec_ray)) * np.real(s)) < 0)
print(f"Raw BER (no MRC): {ber_raw:.4f} (expected ~0.5)")

# BER with correct MRC
mrc = np.conj(h) * rec_ray
ber_mrc = np.mean((np.sign(np.real(mrc)) * np.real(s)) < 0)
print(f"MRC BER: {ber_mrc:.4f} (theory Rayleigh: {0.5*(1-np.sqrt(snr_lin/(1+snr_lin))):.4f})")
