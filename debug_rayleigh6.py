"""
逐步比較 Channel class 與直接數值計算
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from mc_comm_system.channel import Channel

# 固定 4 個符號，完全可追蹤
s = np.array([1.0+0j, -1.0+0j, 1.0+0j, -1.0+0j])
snr_db = 2.0
seed = 42

ch = Channel('Rayleigh', random_seed=seed)
received, state = ch.transmit(s, snr_db, return_channel_state=True)

h = np.array(state.fading_coefficients, dtype=complex)
print("h from state:", h)
print("received:", received)
print("h*s:", h*s)
print("noise (received - h*s):", received - h*s)

# MRC
mrc = np.conj(h) * received
print("MRC output (conj(h)*received):", mrc)
print("Re(MRC):", np.real(mrc))
print("Expected signal (|h|^2 * s):", np.abs(h)**2 * s)
print("Actual noise after MRC:", mrc - np.abs(h)**2 * s)

# 驗證: Re(conj(h)*received) = |h|^2 * Re(s) + Re(conj(h)*noise)
print()
print("=== 理論追蹤 ===")
noise = received - h * s
print("|h|^2:", np.abs(h)**2)
print("Re(conj(h)*noise):", np.real(np.conj(h)*noise))
print("Sum:", np.abs(h)**2 * np.real(s) + np.real(np.conj(h)*noise))
print("Actual Re(MRC):", np.real(mrc))
