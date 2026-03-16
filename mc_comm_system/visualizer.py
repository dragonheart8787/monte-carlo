"""
模組 8：視覺化與報表模組

圖表類型：
- BER vs SNR
- SER vs SNR
- 模擬值 vs 理論值
- 收斂曲線
- Constellation 圖
"""

import warnings
import logging

warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*Font.*default.*")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非互動模式，避免 GUI 阻塞
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Any
from pathlib import Path

from .modulation import Modulator
from .theory import get_theoretical_ber

# 中文字體設定（Windows）
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


class Visualizer:
    """視覺化與報表產生器"""

    def __init__(self, figsize: tuple = (8, 6), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi

    def _ber_for_plot(self, ber: np.ndarray, ci_high: np.ndarray) -> np.ndarray:
        """0-error 點：用 CI 上限估計，避免 log(0)"""
        ber = np.asarray(ber)
        ci_high = np.asarray(ci_high)
        out = np.where(ber < 1e-10, np.maximum(ci_high, 1e-10), ber)
        return out

    def plot_ber_vs_snr(
        self,
        results: Dict[str, Any],
        theoretical: bool = True,
        show_ci: bool = True,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        BER vs SNR：主圖僅 BER vs theory。
        0-error 點用 CI 上限估計標示。
        """
        snr = np.array(results["snr_db"])
        ber = np.array(results["ber"])
        ci_low = np.array(results.get("ber_ci_low", [0] * len(snr)))
        ci_high = np.array(results.get("ber_ci_high", [0] * len(snr)))
        ber_plot = self._ber_for_plot(ber, ci_high)

        cfg = results.get("config", {})
        mod = cfg.get("modulation", "BPSK")
        ch = cfg.get("channel", "AWGN")
        tit = title or f"{mod} over {ch}: Simulated vs Theoretical BER"

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        zero_mask = ber < 1e-10
        non_zero = ~zero_mask

        if np.any(non_zero):
            ax.semilogy(snr[non_zero], ber_plot[non_zero], "b-o", label="Simulated BER", markersize=6)
        if np.any(zero_mask):
            ax.semilogy(
                snr[zero_mask], ber_plot[zero_mask],
                "b^", markersize=8, fillstyle="none", markeredgewidth=2,
                label="No observed errors (upper bound)",
                zorder=5,
            )
            # 對最右側 0-error 點加註記
            idx = np.where(zero_mask)[0][-1]
            ax.annotate(
                "upper bound due to zero observed errors",
                xy=(snr[idx], ber_plot[idx]), xytext=(10, 0), textcoords="offset points",
                fontsize=7, ha="left", color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

        if theoretical:
            theory = get_theoretical_ber(mod, ch, snr, cfg.get("rician_k", 3.0))
            if theory is not None:
                ax.semilogy(snr, np.asarray(theory).flatten(), "r--", label="Theoretical BER")

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Bit Error Rate")
        ax.set_title(tit)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        out = save_path or "ber_vs_snr.png"
        plt.savefig(out)
        plt.close()

    def plot_ber_ci_allocation(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """CI width 與 trial allocation 專圖"""
        snr = np.array(results["snr_db"])
        ci_low = np.array(results.get("ber_ci_low", []))
        ci_high = np.array(results.get("ber_ci_high", []))
        raw = results.get("raw_results", [])
        trials = [r.get("num_trials", 0) for r in raw] if raw else []

        cfg = results.get("config", {})
        mod = cfg.get("modulation", "BPSK")
        ch = cfg.get("channel", "AWGN")
        tit = title or f"{mod} over {ch}: CI Width and Trial Allocation"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 0.8), dpi=self.dpi)

        if len(ci_low) == len(snr) and len(ci_high) == len(snr):
            width = np.array(ci_high) - np.array(ci_low)
            bars = ax1.bar(range(len(snr)), width, color="steelblue", alpha=0.8)
            for b, w in zip(bars, width):
                ax1.text(b.get_x() + b.get_width() / 2, w * 1.15, f"{w:.1e}",
                         ha="center", va="bottom", fontsize=7)
            ax1.set_xticks(range(len(snr)))
            ax1.set_xticklabels([f"{s:.0f}" for s in snr])
            ax1.set_ylabel("CI Width (upper - lower)")
            ax1.set_title("Confidence Interval Width")
            ax1.set_yscale("log")
            ax1.grid(True, alpha=0.3)

        if trials and len(trials) == len(snr):
            ax2.bar(range(len(snr)), trials, color="seagreen", alpha=0.8)
            ax2.set_xticks(range(len(snr)))
            ax2.set_xticklabels([f"{s:.0f}" for s in snr])
            ax2.set_xlabel("SNR (dB)")
            ax2.set_ylabel("Trials")
            ax2.set_title("Trial Allocation")
            ax2.grid(True, alpha=0.3)

        plt.suptitle(tit, fontsize=12)
        plt.tight_layout()
        out = save_path or "ber_ci_allocation.png"
        plt.savefig(out)
        plt.close()

    def plot_constellation(
        self,
        modulation: str = "BPSK",
        received: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """星座圖"""
        mod = Modulator(modulation)
        const = mod.get_constellation()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.scatter(
            np.real(const),
            np.imag(const),
            c="blue",
            s=80,
            label="理論星座",
            zorder=2,
        )

        if received is not None and len(received) > 0:
            ax.scatter(
                np.real(received[:500]),
                np.imag(received[:500]),
                c="red",
                alpha=0.3,
                s=5,
                label="接收訊號",
            )

        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.set_title(f"{modulation} Constellation Diagram")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out = save_path or "constellation.png"
        plt.savefig(out)
        plt.close()

    def plot_convergence(
        self,
        ber_history: List[float],
        true_value: Optional[float] = None,
        save_path: Optional[str] = None,
        snr_db: Optional[float] = None,
        title: Optional[str] = None,
    ) -> None:
        """收斂曲線：running mean vs 樣本數"""
        n = len(ber_history)
        running_mean = np.cumsum(ber_history) / np.arange(1, n + 1)
        final_ber = float(running_mean[-1]) if n > 0 else 0.0

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.semilogy(
            np.arange(1, n + 1),
            running_mean,
            "b-",
            label="Running mean BER",
        )
        if true_value is not None:
            ax.axhline(true_value, color="r", linestyle="--", label="Theoretical")
        ax.set_xlabel("Number of trials")
        ax.set_ylabel("BER")
        tit = title or (f"Monte Carlo Convergence at {snr_db:.0f} dB" if snr_db is not None else "Monte Carlo Convergence")
        ax.set_title(tit)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        # 右上角文字框：Final simulated BER, Theoretical BER, Relative error
        if true_value is not None and true_value > 1e-15:
            rel_err = abs(final_ber - true_value) / true_value if true_value > 0 else 0
            box_text = (
                f"Final simulated BER: {final_ber:.2e}\n"
                f"Theoretical BER: {true_value:.2e}\n"
                f"Relative error: {rel_err:.1%}"
            )
            ax.text(0.98, 0.98, box_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.tight_layout()
        out = save_path or "convergence.png"
        plt.savefig(out)
        plt.close()

    def plot_comparison(
        self,
        results_list: List[Dict[str, Any]],
        labels: List[str],
        save_path: Optional[str] = None,
        subtitle: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        多情境比較圖。observed BER 與 bound 點分開處理：
        - observed: 實線連接
        - no observed errors: 僅 marker，不與主線連接
        避免高 SNR 折線「跳回」的誤導。
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

        for i, (res, lbl) in enumerate(zip(results_list, labels)):
            snr = np.array(res["snr_db"])
            ber = np.array(res["ber"])
            ci_high = np.array(res.get("ber_ci_high", [1] * len(snr)))
            ber_plot = self._ber_for_plot(ber, ci_high)
            obs_mask = ber >= 1e-10
            bound_mask = ber < 1e-10
            c = colors[i]

            # observed BER: 實線連接
            if np.any(obs_mask):
                s_obs = snr[obs_mask]
                b_obs = ber_plot[obs_mask]
                ax.semilogy(s_obs, b_obs, "o-", color=c, label=lbl, markersize=5)

            # bound 點: 僅 marker，不與主線連接
            if np.any(bound_mask):
                s_bnd = snr[bound_mask]
                b_bnd = ber_plot[bound_mask]
                ax.semilogy(
                    s_bnd, b_bnd, "^", color=c, markersize=6,
                    fillstyle="none", markeredgewidth=2,
                    label=None,
                )

        handles, labls = ax.get_legend_handles_labels()
        has_bounds = any(np.any(np.array(res["ber"]) < 1e-10) for res in results_list)
        if has_bounds:
            extra = Line2D([0], [0], marker="^", color="gray", linestyle="None",
                          markersize=6, fillstyle="none", markeredgewidth=2,
                          label="Upper bound (0 errors)")
            handles.append(extra)
            labls.append("Upper bound (0 errors)")
        ax.legend(handles=handles, labels=labls)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Bit Error Rate")
        tit = title or "BER Comparison across Modulation/Channel Conditions"
        ax.set_title(tit)
        sub = subtitle or (
            "Coherent detection, uncoded. Rayleigh: no channel compensation. "
            "Triangles: upper bound (no observed errors), not connected to main curve."
        )
        fig.text(0.5, 0.01, sub, ha="center", fontsize=8, style="italic")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout(rect=[0, 0.06, 1, 1])

        out = save_path or "comparison.png"
        plt.savefig(out)
        plt.close()

    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """產生文字報表摘要"""
        lines = [
            "=" * 50,
            "蒙地卡羅通訊模擬報告",
            "=" * 50,
            "",
        ]
        cfg = results.get("config", {})
        for k, v in cfg.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("SNR (dB) | BER (mean) | CI low | CI high")
        lines.append("-" * 50)
        for r in results.get("raw_results", []):
            lines.append(
                f"  {r['snr_db']:6.1f}  | {r['ber_mean']:.2e} | "
                f"{r['ber_ci_low']:.2e} | {r['ber_ci_high']:.2e}"
            )
        lines.append("")
        report = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")
        return report
