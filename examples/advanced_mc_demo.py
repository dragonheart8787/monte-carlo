"""
進階蒙地卡羅示範

展示：
- CI-based stopping
- Adaptive sample allocation
- 通道拆層（自訂 FadingProcess）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_comm_system import SimulationConfig, MonteCarloEngine, Visualizer


def demo_ci_stopping():
    """CI-based stopping：信賴區間達標即停止"""
    print("\n=== CI-based Stopping ===")
    cfg = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(6, 6),
        num_trials=100,
        bits_per_simulation=1000,
        random_seed=42,
    )
    engine = MonteCarloEngine(cfg)
    r = engine.run_with_ci_stopping(
        snr_db=6.0,
        target_ci_width_relative=0.2,
        min_trials=30,
        max_trials=500,
        batch_size=10,
    )
    print(f"BER: {r['ber_mean']:.2e}, trials: {r['num_trials']}, stopped: {r['ci_stopped']}")


def demo_adaptive():
    """Adaptive allocation：低 BER 區自動增加樣本"""
    print("\n=== Adaptive Sample Allocation ===")
    cfg = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(0, 12),
        snr_db_step=4,
        num_trials=50,
        bits_per_simulation=500,
        random_seed=42,
    )
    engine = MonteCarloEngine(cfg)
    r = engine.run_adaptive(
        snr_points=[0, 4, 8, 12],
        base_trials=30,
        max_trials_per_point=500,
        target_relative_error=0.3,
    )
    print("SNR | BER      | trials")
    for i in range(len(r["snr_db"])):
        print(f"  {r['snr_db'][i]:2.0f} | {r['ber'][i]:.2e} | {r['num_trials_per_point'][i]}")


if __name__ == "__main__":
    demo_ci_stopping()
    demo_adaptive()
