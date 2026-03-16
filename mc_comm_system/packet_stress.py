"""
Packet-level 壓力故事線

完整故事：
- Burst noise 對 BER 與 PER 的差異
- Block fading 為何 PER 惡化更明顯
- 同 BER 下，不同錯誤分布可能導致不同 packet success
"""

from typing import Any, Dict, List

from .config import SimulationConfig
from .monte_carlo_engine import MonteCarloEngine
from .theory import get_theoretical_ber


def run_ber_vs_per_burst_story(
    snr_points: List[float] = None,
    burst_probs: List[float] = None,
    packet_length: int = 100,
    num_trials: int = 80,
    bits_per_simulation: int = 5000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    故事線 1：Burst noise 對 BER 與 PER 的差異。

    當 burst 發生時，單一封包內多 bit 同時錯，PER 惡化幅度 > BER。
    """
    snr_points = snr_points or [4, 6, 8, 10, 12]
    burst_probs = burst_probs or [0.0, 0.02, 0.05]

    story = {"scenarios": [], "summary": ""}

    for bp in burst_probs:
        cfg = SimulationConfig(
            modulation="BPSK",
            channel="AWGN",
            snr_db_range=(min(snr_points), max(snr_points)),
            snr_db_step=2,
            num_trials=num_trials,
            bits_per_simulation=bits_per_simulation,
            packet_length=packet_length,
            burst_noise_prob=bp,
            burst_noise_ratio=10.0,
            random_seed=random_seed,
        )
        engine = MonteCarloEngine(cfg)
        r = engine.run()

        ber_per_snr = []
        per_per_snr = []
        for raw in r.get("raw_results", []):
            ber_per_snr.append(raw.get("ber_mean", 0))
            per_per_snr.append(raw.get("per_mean", 0))

        story["scenarios"].append({
            "burst_prob": bp,
            "snr_db": r["snr_db"],
            "ber": ber_per_snr,
            "per": per_per_snr,
        })

    story["summary"] = (
        "Burst noise 使錯誤集中於少數封包，PER 惡化幅度大於 BER。"
        "burst_prob 越高，PER/BER 比值越大。"
    )
    return story


def run_block_fading_per_story(
    snr_points: List[float] = None,
    packet_length: int = 100,
    block_size: int = 50,
    num_trials: int = 80,
    bits_per_simulation: int = 5000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    故事線 2：Block fading 為何 PER 惡化更明顯。

    Block fading 下，同一 block 內符號共享差通道，錯誤相關性高，
    易導致整封包失敗，PER 比 fast fading 更差。
    """
    snr_points = snr_points or [4, 8, 12, 16]

    results = {}
    for mode in ["fast", "block"]:
        cfg = SimulationConfig(
            modulation="BPSK",
            channel="Rayleigh",
            snr_db_range=(min(snr_points), max(snr_points)),
            snr_db_step=4,
            num_trials=num_trials,
            bits_per_simulation=bits_per_simulation,
            packet_length=packet_length,
            fading_mode=mode,
            block_size=block_size,
            random_seed=random_seed,
        )
        engine = MonteCarloEngine(cfg)
        r = engine.run()
        per_list = [x.get("per_mean", 0) for x in r.get("raw_results", [])]
        ber_list = [x.get("ber_mean", 0) for x in r.get("raw_results", [])]
        results[mode] = {
            "snr_db": r["snr_db"],
            "ber": ber_list,
            "per": per_list,
        }

    return {
        "fast": results["fast"],
        "block": results["block"],
        "summary": (
            "Block fading 下同一 block 錯誤相關性高，"
            "PER 惡化較 fast fading 明顯。"
        ),
    }


def _build_packet_stress_dashboards(suite: Dict[str, Any], output_dir: str) -> None:
    """產出 burst BER/PER 圖與 fast vs block fading PER 圖"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = Path(output_dir)

    # 1. Burst noise BER vs PER
    burst = suite["burst_noise_ber_vs_per"]
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for s in burst["scenarios"]:
        bp = s["burst_prob"]
        ax1.semilogy(s["snr_db"], s["ber"], "o-", label=f"burst_prob={bp}")
        ax2.semilogy(s["snr_db"], s["per"], "o-", label=f"burst_prob={bp}")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("BER")
    ax1.set_title("Burst Noise: BER")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("PER")
    ax2.set_title("Burst Noise: PER")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    fig1.suptitle("Burst Noise: BER vs PER (PER degrades more than BER)")
    plt.tight_layout()
    plt.savefig(out / "burst_ber_per_dashboard.png")
    plt.close()

    # 2. Fast vs Block fading PER
    block = suite["block_fading_per_degradation"]
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(block["fast"]["snr_db"], block["fast"]["per"], "o-", label="Fast fading")
    ax.semilogy(block["block"]["snr_db"], block["block"]["per"], "s-", label="Block fading")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PER")
    ax.set_title("Fast vs Block Fading: Block PER degrades more")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "block_fading_dashboard.png")
    plt.close()

    # 3. Conclusions text
    concl = [
        "=== Packet Stress Conclusions ===",
        "",
        "1. Burst noise: Errors cluster in few packets -> PER/BER ratio increases.",
        "2. Block fading: Same-block errors correlated -> PER worse than fast fading.",
        "3. Same BER, different error distribution -> different packet success.",
    ]
    (out / "packet_stress_conclusions.txt").write_text("\n".join(concl))


def run_packet_stress_suite(
    output_path: str = None,
    output_dir: str = "packet_stress_results",
    save_dashboards: bool = True,
) -> Dict[str, Any]:
    """
    執行完整 packet-level 壓力故事線，產出可展示結果。
    產出：JSON、burst BER/PER 圖、fast vs block PER 圖、結論檔。
    """
    from pathlib import Path

    burst_story = run_ber_vs_per_burst_story()
    block_story = run_block_fading_per_story()

    suite = {
        "burst_noise_ber_vs_per": burst_story,
        "block_fading_per_degradation": block_story,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = Path(output_path) if output_path else out / "packet_stress_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        __import__("json").dumps(suite, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if save_dashboards:
        _build_packet_stress_dashboards(suite, str(out))

    return suite
