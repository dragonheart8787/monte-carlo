import numpy as np
from mc_comm_system.channel import Channel

rng = np.random.default_rng(999)
N = 100000
bits = rng.integers(0, 2, N)
s = np.where(bits == 1, 1.0+0j, -1.0+0j)

ch = Channel('Rayleigh', random_seed=42)
snr_db = 2.0
received, state = ch.transmit(s, snr_db, return_channel_state=True)
h = np.array(state.fading_coefficients, dtype=complex)

print("Signal statistics:")
print(f"  |s| mean: {np.mean(np.abs(s)):.4f}")
print(f"  |h| mean: {np.mean(np.abs(h)):.4f}, |h|^2 mean: {np.mean(np.abs(h)**2):.4f}")
print(f"  |received| mean: {np.mean(np.abs(received)):.4f}")

# MRC
mrc = received * np.conj(h)
re_mrc = np.real(mrc)
print(f"\nMRC output:")
print(f"  re_mrc mean: {np.mean(re_mrc):.4f} (expected ~0 for 50/50 +1/-1)")
print(f"  re_mrc std: {np.std(re_mrc):.4f}")
print(f"  P(re_mrc > 0 | s=+1): {np.mean(re_mrc[bits==1] > 0):.4f} (should be close to 0.89)")
print(f"  P(re_mrc < 0 | s=-1): {np.mean(re_mrc[bits==0] < 0):.4f} (should be close to 0.89)")

errors = np.sign(re_mrc) != np.real(s)
print(f"\nActual BER: {np.mean(errors):.4f} (theory: 0.1085)")

# Check raw received re vs s
print(f"\nRaw received (no MRC):")
print(f"  P(re_raw > 0 | s=+1): {np.mean(np.real(received[bits==1]) > 0):.4f}")
