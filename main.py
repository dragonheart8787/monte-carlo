"""
蒙地卡羅通訊效能評估系統 - 主程式

示範三種使用情境：
1. 基礎版：BPSK + AWGN，BER vs SNR
2. 完整版：多調變、多通道比較
3. 研究版：收斂監控、變異數縮減
"""

import warnings
import logging

warnings.filterwarnings("ignore", message=".*glyph.*")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
from mc_comm_system import (
    SimulationConfig,
    MonteCarloEngine,
    Visualizer,
)


def run_basic_demo():
    """基礎版：BPSK + AWGN"""
    print("\n" + "=" * 60)
    print("【基礎版】BPSK + AWGN 通道")
    print("=" * 60)

    config = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(0, 12),
        snr_db_step=2,
        bits_per_simulation=5000,
        num_trials=50,
        random_seed=42,
    )

    engine = MonteCarloEngine(config)
    results = engine.run(high_snr_boost=True, high_snr_threshold_db=8, high_snr_multiplier=3)

    viz = Visualizer()
    viz.plot_ber_vs_snr(results, theoretical=True, save_path="ber_vs_snr.png")
    viz.plot_ber_ci_allocation(results, save_path="ber_ci_allocation.png")
    print(viz.generate_report(results))


def run_complete_demo():
    """完整版：多調變、多通道比較"""
    print("\n" + "=" * 60)
    print("【完整版】多調變與通道比較")
    print("=" * 60)

    scenarios = [
        ("BPSK + AWGN", "BPSK", "AWGN"),
        ("QPSK + AWGN", "QPSK", "AWGN"),
        ("BPSK + Rayleigh", "BPSK", "Rayleigh"),
    ]

    results_list = []
    labels = []

    for name, mod, ch in scenarios:
        config = SimulationConfig(
            modulation=mod,
            channel=ch,
            snr_db_range=(0, 16),
            snr_db_step=2,
            bits_per_simulation=3000,
            num_trials=30,
            random_seed=42,
        )
        engine = MonteCarloEngine(config)
        results = engine.run()
        results_list.append(results)
        labels.append(name)

    viz = Visualizer()
    viz.plot_comparison(
        results_list, labels,
        subtitle="Coherent, uncoded. 3000 bits/sim, 30 trials. Rayleigh: no channel compensation. Triangles: upper bound (no observed errors).",
    )


def run_research_demo():
    """研究版：收斂監控"""
    print("\n" + "=" * 60)
    print("【研究版】收斂監控示範")
    print("=" * 60)

    config = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(6, 6),  # 單點
        bits_per_simulation=1000,
        num_trials=500,
        random_seed=42,
    )

    engine = MonteCarloEngine(config)
    # 手動收集收斂歷史
    ber_history = []
    for _ in range(200):
        r = engine._run_single_trial(6.0, 1000, None)
        ber_history.append(r["ber"])

    # 理論值 (BPSK AWGN @ 6dB)
    from scipy import special
    snr_linear = 10 ** (6 / 10)
    theory_ber = 0.5 * special.erfc(np.sqrt(snr_linear))

    viz = Visualizer()
    viz.plot_convergence(ber_history, true_value=theory_ber)
    print(f"理論 BER @ 6dB: {theory_ber:.6e}")
    print(f"最終模擬 BER: {np.mean(ber_history):.6e}")


if __name__ == "__main__":
    print("蒙地卡羅通訊效能評估系統")
    print("請選擇執行模式：")
    print("  1. 基礎版 (BPSK + AWGN)")
    print("  2. 完整版 (多調變/通道比較)")
    print("  3. 研究版 (收斂監控)")
    print("  4. 全部執行")

    try:
        choice = input("輸入選項 (1-4，預設 1): ").strip() or "1"
    except EOFError:
        choice = "1"

    if choice == "1":
        run_basic_demo()
    elif choice == "2":
        run_complete_demo()
    elif choice == "3":
        run_research_demo()
    elif choice == "4":
        run_basic_demo()
        run_complete_demo()
        run_research_demo()
    else:
        print("無效選項，執行基礎版")
        run_basic_demo()
