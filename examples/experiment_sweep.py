"""
實驗管理層示範：批次 sweep modulation × channel

執行：python examples/experiment_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_comm_system import SimulationConfig, ExperimentManager, Visualizer


def main():
    mgr = ExperimentManager(output_root="experiments", prefix="exp")

    # 批次 sweep
    records = mgr.run_sweep(
        modulations=["BPSK", "QPSK"],
        channels=["AWGN", "Rayleigh"],
        base_config=SimulationConfig(
            snr_db_range=(0, 10),
            snr_db_step=2,
            bits_per_simulation=2000,
            num_trials=20,
            random_seed=42,
        ),
    )

    print(mgr.generate_summary_table(output_path="experiments/summary.txt"))

    # 繪製比較圖
    viz = Visualizer()
    results_list = [r.results for r in records]
    labels = [f"{r.config_snapshot['modulation']}+{r.config_snapshot['channel']}" for r in records]
    viz.plot_comparison(results_list, labels, save_path="experiments/comparison.png")


if __name__ == "__main__":
    main()
