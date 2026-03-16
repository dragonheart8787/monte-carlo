"""
完整報告產生器

執行全部實驗（主程式、Benchmark、Packet Stress），產出整合報告與圖表。
"""

import warnings
import logging

warnings.filterwarnings("ignore", message=".*glyph.*")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

# 確保專案根目錄在 path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from mc_comm_system import (
    SimulationConfig,
    MonteCarloEngine,
    Visualizer,
)
from mc_comm_system.benchmark_suite import run_benchmark_suite
from mc_comm_system.packet_stress import run_packet_stress_suite


def run_main_demos():
    """主程式三模式"""
    Path("report_figures").mkdir(exist_ok=True)

    # 基礎版
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
    basic_results = engine.run(high_snr_boost=True, high_snr_threshold_db=8, high_snr_multiplier=3)
    viz = Visualizer()
    viz.plot_ber_vs_snr(basic_results, theoretical=True, save_path="report_figures/ber_vs_snr.png")
    viz.plot_ber_ci_allocation(basic_results, save_path="report_figures/ber_ci_allocation.png")

    # 完整版
    scenarios = [
        ("BPSK + AWGN", "BPSK", "AWGN"),
        ("QPSK + AWGN", "QPSK", "AWGN"),
        ("BPSK + Rayleigh", "BPSK", "Rayleigh"),
    ]
    results_list, labels = [], []
    for name, mod, ch in scenarios:
        cfg = SimulationConfig(
            modulation=mod, channel=ch,
            snr_db_range=(0, 16), snr_db_step=2,
            bits_per_simulation=3000, num_trials=30, random_seed=42,
        )
        r = MonteCarloEngine(cfg).run()
        results_list.append(r)
        labels.append(name)
    viz.plot_comparison(
        results_list, labels,
        save_path="report_figures/comparison.png",
        subtitle="Coherent, uncoded. 3000 bits/sim, 30 trials. Rayleigh: no channel compensation. Triangles: upper bound (no observed errors).",
    )

    # 研究版
    from scipy import special
    engine = MonteCarloEngine(SimulationConfig(
        modulation="BPSK", channel="AWGN",
        snr_db_range=(6, 6), bits_per_simulation=1000, num_trials=500, random_seed=42,
    ))
    ber_history = [engine._run_single_trial(6.0, 1000, None)["ber"] for _ in range(200)]
    theory_ber = 0.5 * special.erfc(np.sqrt(10 ** (6 / 10)))
    viz.plot_convergence(
        ber_history, true_value=theory_ber,
        save_path="report_figures/convergence.png",
        snr_db=6.0,
    )

    return {
        "basic": basic_results,
        "labels": labels,
        "theory_ber": theory_ber,
        "simulated_ber": np.mean(ber_history),
    }


def build_report_txt(main_data, benchmark_result, packet_stress_data):
    """組裝完整報告文字"""
    lines = [
        "=" * 70,
        "蒙地卡羅通訊效能評估系統 — 完整報告",
        "=" * 70,
        f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "一、主程式實驗結果",
        "-" * 50,
        "",
        "【基礎版】BPSK + AWGN",
        "  SNR (dB) | BER (mean) | CI low | CI high",
        "  " + "-" * 50,
    ]
    for r in main_data["basic"].get("raw_results", []):
        lines.append(
            f"  {r['snr_db']:6.1f}  | {r['ber_mean']:.2e} | "
            f"{r['ber_ci_low']:.2e} | {r['ber_ci_high']:.2e}"
        )
    lines.extend([
        "",
        "【完整版】多調變比較",
        f"  情境：{', '.join(main_data['labels'])}",
        "",
        "【研究版】收斂監控",
        f"  理論 BER @ 6dB: {main_data['theory_ber']:.6e}",
        f"  模擬 BER: {main_data['simulated_ber']:.6e}",
        "",
        "二、Benchmark Suite",
        "-" * 50,
        f"  Passed: {benchmark_result['passed']}",
        f"  Failed: {benchmark_result['failed']}",
        f"  All passed: {benchmark_result['all_passed']}",
        "",
        "三、Packet Stress",
        "-" * 50,
        f"  Burst: {packet_stress_data.get('burst_noise_ber_vs_per', {}).get('summary', 'N/A')}",
        f"  Block fading: {packet_stress_data.get('block_fading_per_degradation', {}).get('summary', 'N/A')}",
        "",
        "四、產出目錄",
        "-" * 50,
        "  report_figures/     — BER、比較、收斂圖",
        "  benchmark_results/  — Benchmark 總覽",
        "  packet_stress_results/ — Burst/Block 圖",
        "",
        "=" * 70,
    ])
    return "\n".join(lines)


def main():
    print("蒙地卡羅通訊效能評估系統 — 完整報告產生器")
    print("=" * 60)

    Path("docs").mkdir(exist_ok=True)

    # 1. 主程式
    print("\n[1/3] 主程式（基礎 / 完整 / 研究）...")
    main_data = run_main_demos()

    # 2. Benchmark
    print("[2/3] Benchmark Suite...")
    benchmark_result = run_benchmark_suite(
        output_root="benchmark_results",
        save_figures=True,
    )

    # 3. Packet Stress
    print("[3/3] Packet Stress...")
    packet_stress_data = run_packet_stress_suite(
        output_dir="packet_stress_results",
        save_dashboards=True,
    )

    # 產出報告
    report_txt = build_report_txt(main_data, benchmark_result, packet_stress_data)
    out_txt = Path("docs/完整報告.txt")
    out_txt.write_text(report_txt, encoding="utf-8")

    # 複製關鍵圖到報告目錄
    report_img = Path("docs/images")
    report_img.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        ("report_figures/ber_vs_snr.png", "report_ber_vs_snr.png"),
        ("report_figures/ber_ci_allocation.png", "report_ber_ci_allocation.png"),
        ("report_figures/comparison.png", "report_comparison.png"),
        ("report_figures/convergence.png", "report_convergence.png"),
        ("benchmark_results/benchmark_summary_dashboard.png", "report_benchmark.png"),
        ("packet_stress_results/burst_ber_per_dashboard.png", "report_burst.png"),
        ("packet_stress_results/block_fading_dashboard.png", "report_block_fading.png"),
    ]:
        if Path(src).exists():
            shutil.copy(src, report_img / dst)

    print(f"\n報告已產出：{out_txt}")
    print("圖表：report_figures/、benchmark_results/、packet_stress_results/")
    print("All passed:", benchmark_result["all_passed"])
    print("\n完成。")


if __name__ == "__main__":
    main()
