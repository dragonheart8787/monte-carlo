"""
快速執行範例 - 無需互動輸入

直接執行：python examples/quick_run.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_comm_system import SimulationConfig, MonteCarloEngine, Visualizer


def main():
    config = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(0, 10),
        snr_db_step=2,
        bits_per_simulation=3000,
        num_trials=30,
        random_seed=42,
    )

    engine = MonteCarloEngine(config)
    results = engine.run()

    viz = Visualizer()
    viz.plot_ber_vs_snr(results, theoretical=True, show_ci=True)
    print(viz.generate_report(results))


if __name__ == "__main__":
    main()
