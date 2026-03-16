"""
官方基準實驗組合

固定定義的標準案例，供每次改版回歸驗證。
確保不是越改越歪。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SimulationConfig
from .experiment_manager import ExperimentManager, ExperimentRecord
from .theory import get_theoretical_ber, relative_error


@dataclass
class BenchmarkCase:
    """單一基準案例定義"""
    name: str
    config: SimulationConfig
    description: str
    # 驗證條件（可選）
    max_relative_error: Optional[float] = None
    min_trials: Optional[int] = None


# 官方基準案例
BENCHMARK_CASES = [
    BenchmarkCase(
        name="BPSK_AWGN_theory_validation",
        config=SimulationConfig(
            modulation="BPSK",
            channel="AWGN",
            snr_db_range=(0, 12),
            snr_db_step=2,
            num_trials=100,
            bits_per_simulation=2000,
            random_seed=42,
            experiment_type="validation",
            experiment_rigor="high",
            theory_available=True,
        ),
        description="BPSK + AWGN 理論驗證",
        max_relative_error=0.5,
        min_trials=50,
    ),
    BenchmarkCase(
        name="QPSK_Rayleigh_convergence",
        config=SimulationConfig(
            modulation="QPSK",
            channel="Rayleigh",
            snr_db_range=(0, 14),
            snr_db_step=2,
            num_trials=80,
            bits_per_simulation=1500,
            random_seed=42,
            experiment_type="validation",
            experiment_rigor="baseline",
        ),
        description="QPSK + Rayleigh 收斂驗證",
        min_trials=50,
    ),
    BenchmarkCase(
        name="Burst_noise_PER_stress",
        config=SimulationConfig(
            modulation="BPSK",
            channel="AWGN",
            snr_db_range=(4, 12),
            snr_db_step=2,
            num_trials=100,
            bits_per_simulation=1000,
            packet_length=100,
            burst_noise_prob=0.05,
            burst_noise_ratio=10.0,
            random_seed=42,
            experiment_type="stress",
            experiment_rigor="baseline",
        ),
        description="Burst noise + PER 壓力測試",
        min_trials=50,
    ),
    BenchmarkCase(
        name="Phase_offset_compensation",
        config=SimulationConfig(
            modulation="BPSK",
            channel="AWGN",
            snr_db_range=(6, 6),
            snr_db_step=2,
            num_trials=150,
            bits_per_simulation=1500,
            phase_offset_rad=0.5,
            random_seed=42,
            experiment_type="comparison",
            experiment_rigor="baseline",
        ),
        description="Phase offset 補償比較（需搭配補償實驗）",
        min_trials=80,
    ),
]


def _build_benchmark_summary_dashboard(
    results: Dict[str, Dict],
    output_path: str,
) -> None:
    """產出四案總覽 dashboard 圖"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, 4))

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        rec = data["record"]
        res = rec.results
        passed = data["passed"]
        snr = np.array(res.get("snr_db", []))
        ber = np.array(res.get("ber", []))
        ax.semilogy(snr, ber, "o-", color=colors[idx], markersize=5)
        if res.get("theory_ber"):
            ax.semilogy(snr, res["theory_ber"], "r--", alpha=0.7, label="theory")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.set_title(f"{name}\n{'PASS' if passed else 'FAIL'}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    plt.suptitle("Benchmark Suite Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_benchmark_suite(
    output_root: str = "benchmark_results",
    save_figures: bool = False,
) -> Dict[str, Any]:
    """
    執行完整基準實驗組合。
    回傳各案例結果與通過/失敗摘要。
    自動產出 benchmark_summary.txt 與 benchmark_summary_dashboard.png。
    """
    mgr = ExperimentManager(output_root=output_root, prefix="bm")
    results = {}
    passed = []
    failed = []

    for case in BENCHMARK_CASES:
        case.config.validate()
        record = mgr.run_single(
            case.config,
            experiment_id=case.name,
            save_results=True,
            save_figures=save_figures,
            description=case.description,
        )

        ok = True
        res = record.results
        cfg = record.config_snapshot

        if case.min_trials and res.get("raw_results"):
            total_trials = sum(r.get("num_trials", 0) for r in res["raw_results"])
            if total_trials < case.min_trials * len(res.get("snr_db", [1])):
                ok = False

        if case.max_relative_error and res.get("theory_ber") and res.get("ber"):
            for i, (sim, th) in enumerate(zip(res["ber"], res["theory_ber"])):
                if th and th > 1e-10:
                    # 跳過極低理論區：模擬 BER=0 時 relative error 無意義
                    if th < 1e-5:
                        continue
                    re = abs(sim - th) / th
                    if re > case.max_relative_error:
                        ok = False
                        break

        results[case.name] = {"record": record, "passed": ok}
        if ok:
            passed.append(case.name)
        else:
            failed.append(case.name)

    out_dir = Path(output_root)
    summary_txt = mgr.generate_summary_table(
        [r["record"] for r in results.values()],
        output_path=out_dir / "benchmark_summary.txt",
    )

    # Benchmark summary dashboard
    dashboard_path = out_dir / "benchmark_summary_dashboard.png"
    _build_benchmark_summary_dashboard(results, str(dashboard_path))

    return {
        "results": results,
        "passed": passed,
        "failed": failed,
        "all_passed": len(failed) == 0,
        "summary_table": summary_txt,
        "summary_dashboard_path": str(dashboard_path),
    }
