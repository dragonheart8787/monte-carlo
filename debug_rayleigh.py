import numpy as np
from mc_comm_system import SimulationConfig, MonteCarloEngine
from mc_comm_system.theory import ber_bpsk_rayleigh
from mc_comm_system.channel import Channel

# 直接測試 MRC 等化效果
print("=== 直接通道測試 ===")
ch = Channel('Rayleigh', random_seed=42)
np.random.seed(0)
s = np.array([1.0+0j]*500 + [-1.0+0j]*500)  # 1000 BPSK symbols

received, state = ch.transmit(s, 2.0, return_channel_state=True)
h = np.array(state.fading_coefficients)
mrc = received * np.conj(h)
decisions = np.sign(np.real(mrc))
errors = np.sum(decisions != np.real(s))
print(f"  MRC BER at SNR=2dB: {errors/1000:.4f} (theory: {ber_bpsk_rayleigh(2):.4f})")

# 多次試驗
print("\n=== 5 次 trial 測試 ===")
cfg = SimulationConfig(modulation='BPSK', channel='Rayleigh',
    snr_db_range=(2, 2), bits_per_simulation=5000, num_trials=5, random_seed=42)
engine = MonteCarloEngine(cfg)
print(f"  use_fading_csi: {engine._use_fading_csi}")
print(f"  compensator: {type(engine.receiver.compensator).__name__}")

for i in range(5):
    result = engine._run_single_trial(2.0, 5000, None)
    print(f"  Trial {i}: BER={result['ber']:.4f}")

print("\n=== SNR 掃描 ===")
for snr in [0, 2, 4, 6, 8]:
    r = engine.run_snr_point(snr, num_trials=30)
    print(f"  SNR={snr:2d}dB | sim BER={r['ber_mean']:.4e} | theory={ber_bpsk_rayleigh(snr):.4e}")
