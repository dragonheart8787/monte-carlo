"""
一頁式 Dashboard 報表

單頁總覽輸出，含：
- BER curve + theory overlay + CI band
- Trial allocation
- Diagnostics summary
- Runtime
- Key config snapshot
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, Optional

from .theory import get_theoretical_ber
from .diagnostics import diagnose_results, format_diagnostics


def _format_diagnostics_ascii(diag: list) -> str:
    """ASCII-only diagnostics for dashboard (avoid font issues)"""
    if not diag:
        return "No diagnostics"
    lines = ["Diagnostics:"]
    for d in diag:
        prefix = "[WARN]" if d["level"] == "warning" else "[INFO]"
        lines.append(f"  {prefix} {d['code']}")
    return "\n".join(lines)


def build_dashboard(
    results: Dict[str, Any],
    output_path: str = "dashboard.png",
    figsize: tuple = (12, 10),
    dpi: int = 120,
) -> str:
    """
    產出一頁式 dashboard 圖檔。
    固定包含：BER 曲線、理論疊加、CI band、trial 分配、診斷、runtime、config 摘要。
    """
    cfg = results.get("config", results.get("config_snapshot", {}))
    snr = np.array(results.get("snr_db", []))
    ber = np.array(results.get("ber", []))
    ci_low = np.array(results.get("ber_ci_low", results.get("confidence_interval_low", [])))
    ci_high = np.array(results.get("ber_ci_high", results.get("confidence_interval_high", [])))
    trials = results.get("num_trials_per_point", [])
    if not trials and results.get("raw_results"):
        trials = [r.get("num_trials", 0) for r in results["raw_results"]]

    diag = diagnose_results(results, cfg)
    diag_text = _format_diagnostics_ascii(diag) if diag else "No diagnostics"

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # 0-error: use CI upper for log scale
    ber_plot = np.where(np.array(ber) < 1e-10, np.maximum(np.array(ci_high), 1e-10), np.array(ber))
    zero_mask = np.array(ber) < 1e-10

    # 1. BER vs theory only (no CI band)
    ax1 = fig.add_subplot(gs[0, :])
    mod, ch = cfg.get("modulation", "BPSK"), cfg.get("channel", "AWGN")
    if np.any(~zero_mask):
        ax1.semilogy(snr[~zero_mask], ber_plot[~zero_mask], "b-o", label="Simulated BER", markersize=6)
    if np.any(zero_mask):
        ax1.semilogy(snr[zero_mask], ber_plot[zero_mask], "b^", markersize=8, fillstyle="none",
                     label="No observed errors (upper bound)", zorder=5)
    theory = get_theoretical_ber(mod, ch, snr, cfg.get("rician_k", 3.0))
    if theory is not None:
        ax1.semilogy(snr, np.asarray(theory).flatten(), "r--", label="Theoretical BER")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("BER")
    ax1.set_title(f"{mod} over {ch}: Simulated vs Theoretical BER")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # 2. CI width
    ax2 = fig.add_subplot(gs[1, 0])
    if len(ci_low) == len(snr) and len(ci_high) == len(snr):
        width = np.array(ci_high) - np.array(ci_low)
        ax2.bar(range(len(snr)), width, color="steelblue", alpha=0.8)
        ax2.set_xticks(range(len(snr)))
        ax2.set_xticklabels([f"{s:.0f}" for s in snr])
        ax2.set_ylabel("CI Width")
        ax2.set_title("Confidence Interval Width")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No CI data", ha="center", va="center")
        ax2.set_title("CI Width")

    # 3. Trial allocation
    ax3_alloc = fig.add_subplot(gs[1, 1])
    if trials and len(trials) == len(snr):
        ax3_alloc.bar(range(len(snr)), trials, color="seagreen", alpha=0.8)
        ax3_alloc.set_xticks(range(len(snr)))
        ax3_alloc.set_xticklabels([f"{s:.0f}" for s in snr])
        ax3_alloc.set_xlabel("SNR (dB)")
        ax3_alloc.set_ylabel("Trials")
        ax3_alloc.set_title("Trial Allocation")
        ax3_alloc.grid(True, alpha=0.3)
    else:
        ax3_alloc.text(0.5, 0.5, "No trial data", ha="center", va="center")
        ax3_alloc.set_title("Trial Allocation")

    # 4. Config snapshot
    ax3 = fig.add_subplot(gs[2, 0])
    config_lines = [
        f"modulation: {cfg.get('modulation', 'N/A')}",
        f"channel: {cfg.get('channel', 'N/A')}",
        f"num_trials: {cfg.get('num_trials', 'N/A')}",
        f"bits/sim: {cfg.get('bits_per_simulation', 'N/A')}",
        f"runtime: {results.get('runtime_sec', 0):.1f}s",
        f"exp_type: {cfg.get('experiment_type', 'N/A')}",
        f"theory_avail: {cfg.get('theory_available', 'N/A')}",
    ]
    ax3.text(0.05, 0.95, "\n".join(config_lines), transform=ax3.transAxes,
             fontsize=9, verticalalignment="top", family="monospace")
    ax3.axis("off")
    ax3.set_title("Config Snapshot")

    # 5. Diagnostics
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.text(0.02, 0.98, diag_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    ax4.axis("off")
    ax4.set_title("Diagnostics")

    plt.suptitle("Monte Carlo Comm Sim - Dashboard", fontsize=14)
    plt.savefig(output_path)
    plt.close()
    return output_path
