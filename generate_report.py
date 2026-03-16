"""
產生完整報告

執行全部實驗、收集結果，產出 docs/完整報告.txt 與更新 docs/完整報告.md
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
from pathlib import Path
from datetime import datetime

from mc_comm_system import (
    SimulationConfig,
    MonteCarloEngine,
    Visualizer,
)


def run_and_collect_basic():
    """基礎版：BPSK + AWGN"""
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
    results = engine.run()
    viz = Visualizer()
    viz.plot_ber_vs_snr(results, theoretical=True, show_ci=True, save_path="report_figures/ber_vs_snr.png")
    return results


def run_and_collect_complete():
    """完整版：多調變與通道比較"""
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
    viz.plot_comparison(results_list, labels, save_path="report_figures/comparison.png")
    return results_list, labels


def run_and_collect_research():
    """研究版：收斂監控"""
    config = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(6, 6),
        bits_per_simulation=1000,
        num_trials=500,
        random_seed=42,
    )
    engine = MonteCarloEngine(config)
    ber_history = []
    for _ in range(200):
        r = engine._run_single_trial(6.0, 1000, None)
        ber_history.append(r["ber"])
    snr_linear = 10 ** (6 / 10)
    from scipy import special
    theory_ber = 0.5 * special.erfc(np.sqrt(snr_linear))
    viz = Visualizer()
    viz.plot_convergence(ber_history, true_value=theory_ber, save_path="report_figures/convergence.png")
    return {"theory": theory_ber, "simulated": np.mean(ber_history), "history": ber_history}


def build_report_text(basic_results, complete_data, research_data):
    """組裝完整報告文字"""
    rl, labels = complete_data
    res = research_data

    lines = [
        "=" * 70,
        "蒙地卡羅通訊效能評估系統 — 完整報告",
        "=" * 70,
        f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "一、摘要",
        "-" * 50,
        "以蒙地卡羅方法為核心的數位通訊鏈路效能評估系統，支援多種調變、通道、",
        "擾動與自適應抽樣，具備理論驗證、實驗管理與 packet-level 分析能力。",
        "",
        "二、實驗結果",
        "-" * 50,
        "",
        "【基礎版】BPSK + AWGN 通道",
        "  配置：SNR 0~12 dB, 5000 bits/sim, 50 trials",
        "",
        "  SNR (dB) | BER (mean) | CI low | CI high",
        "  " + "-" * 50,
    ]

    for r in basic_results.get("raw_results", []):
        lines.append(
            f"  {r['snr_db']:6.1f}  | {r['ber_mean']:.2e} | "
            f"{r['ber_ci_low']:.2e} | {r['ber_ci_high']:.2e}"
        )

    lines.extend([
        "",
        "【完整版】多調變與通道比較",
        f"  情境：{', '.join(labels)}",
        "  產出：report_figures/comparison.png",
        "",
        "【研究版】收斂監控",
        f"  理論 BER @ 6dB: {res['theory']:.6e}",
        f"  最終模擬 BER: {res['simulated']:.6e}",
        f"  相對誤差: {abs(res['simulated'] - res['theory']) / res['theory'] * 100:.2f}%",
        "",
        "三、圖表產出",
        "-" * 50,
        "  report_figures/ber_vs_snr.png   — BER vs SNR (基礎版)",
        "  report_figures/comparison.png   — 多情境比較",
        "  report_figures/convergence.png  — 收斂曲線",
        "",
        "=" * 70,
    ])
    return "\n".join(lines)


def main():
    print("蒙地卡羅通訊效能評估系統 — 完整報告產生器")
    print("=" * 50)

    Path("report_figures").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)

    print("\n[1/3] 執行基礎版 (BPSK + AWGN)...")
    basic_results = run_and_collect_basic()

    print("[2/3] 執行完整版 (多調變/通道比較)...")
    complete_data = run_and_collect_complete()

    print("[3/3] 執行研究版 (收斂監控)...")
    research_data = run_and_collect_research()

    report_text = build_report_text(basic_results, complete_data, research_data)

    out_txt = Path("docs/完整報告.txt")
    out_txt.write_text(report_text, encoding="utf-8")
    print(f"\n報告已產出：{out_txt}")

    print("\n完成。")


if __name__ == "__main__":
    main()
