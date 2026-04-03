import numpy as np
from mc_comm_system.channel import Channel
from mc_comm_system.theory import ber_bpsk_rayleigh

ch = Channel('Rayleigh', random_seed=42)
s = np.array([1.0+0j]*500 + [-1.0+0j]*500)

received, state = ch.transmit(s, 2.0, return_channel_state=True)
h = np.array(state.fading_coefficients)

mrc = received * np.conj(h)
re_mrc = np.real(mrc)

print("=== 診斷 MRC 等化 ===")
print(f"  h[:5] magnitudes: {np.abs(h[:5])}")
print(f"  |h|^2 mean: {np.mean(np.abs(h)**2):.4f}")
print(f"  re_mrc[:5]: {re_mrc[:5]}")
print(f"  transmitted s real[:5]: {np.real(s[:5])}")

# 符號層面看
correct = np.sum(np.sign(re_mrc) == np.real(s))
print(f"  Correct decisions: {correct}/1000")

# 看一下沒有 MRC 的情況（直接 received）
re_raw = np.real(received)
correct_raw = np.sum(np.sign(re_raw) == np.real(s))
print(f"\n  Without MRC - Correct decisions: {correct_raw}/1000")

# 統計 |h|² 分布
print(f"\n  |h|^2 stats: min={np.min(np.abs(h)**2):.3f}, max={np.max(np.abs(h)**2):.3f}")
print(f"  P(|h|^2 < 0.01) = {np.mean(np.abs(h)**2 < 0.01):.3f}")

# 看哪些點 MRC 判錯了
errors_mrc = np.sign(re_mrc) != np.real(s)
print(f"\n  MRC errors: {np.sum(errors_mrc)}")
if np.any(errors_mrc):
    idx = np.where(errors_mrc)[0][:5]
    for i in idx:
        print(f"    idx={i}: s={np.real(s[i]):.0f}, h={h[i]:.3f}, |h|²={np.abs(h[i])**2:.4f}, re_mrc={re_mrc[i]:.4f}")

# 純 AWGN 比較
ch_awgn = Channel('AWGN', random_seed=42)
rec_awgn = ch_awgn.transmit(s, 2.0)
correct_awgn = np.sum(np.sign(np.real(rec_awgn)) == np.real(s))
print(f"\n  AWGN without fading: correct={correct_awgn}/1000")
print(f"  AWGN expected errors (theory): {ber_bpsk_rayleigh(2)*1000:.0f}")
