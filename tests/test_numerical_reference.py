"""
Reference-grade 數值驗證測試

驗證數值可信度，而非僅行為正確。
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_comm_system import (
    SimulationConfig,
    MonteCarloEngine,
    get_theoretical_ber,
)


def test_bpsk_awgn_ber_within_tolerance():
    """
    固定 seed 下，BPSK/AWGN 在多個 SNR 點與理論值誤差不超過門檻。
    相對誤差 < 0.5（50%），或絕對誤差 < 0.01。
    """
    cfg = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(2, 10),
        snr_db_step=2,
        num_trials=200,
        bits_per_simulation=2000,
        random_seed=42,
    )
    engine = MonteCarloEngine(cfg)
    results = engine.run()

    theory = get_theoretical_ber("BPSK", "AWGN", results["snr_db"])
    assert theory is not None

    for i, snr in enumerate(results["snr_db"]):
        sim_ber = results["ber"][i]
        th_ber = float(theory[i])
        if th_ber > 1e-6:
            rel_err = abs(sim_ber - th_ber) / th_ber
            assert rel_err < 0.5, "相對誤差應 < 50%"
        abs_err = abs(sim_ber - th_ber)
        assert abs_err < 0.02, f"SNR={snr}dB 絕對誤差應 < 0.02"


def test_adaptive_high_snr_more_trials():
    """
    Adaptive allocation 下，高 SNR 點 trial 數應顯著高於低 SNR 點。
    （低 BER 需更多樣本才能估計）
    """
    cfg = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(0, 12),
        snr_db_step=6,
        num_trials=50,
        bits_per_simulation=500,
        random_seed=42,
    )
    engine = MonteCarloEngine(cfg)
    r = engine.run_adaptive(
        snr_points=[0, 6, 12],
        base_trials=30,
        max_trials_per_point=800,
        target_relative_error=0.4,
        use_theory=True,
    )
    trials = r["num_trials_per_point"]
    assert trials[2] >= trials[0] * 1.2, "高 SNR 應有較多 trials"


def test_ci_stopping_not_early_in_high_variance():
    """
    CI stopping 在高變異區（低 SNR、高 BER）不應過早停止。
    至少應跑 min_trials。
    """
    cfg = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(0, 0),
        num_trials=100,
        bits_per_simulation=500,
        random_seed=42,
    )
    engine = MonteCarloEngine(cfg)
    r = engine.run_with_ci_stopping(
        snr_db=0.0,
        target_ci_width_relative=0.5,
        min_trials=80,
        max_trials=1000,
        batch_size=10,
    )
    assert r["num_trials"] >= 80, "高變異區應至少跑 min_trials"


def test_empirical_adaptive_produces_valid_result():
    """run_adaptive_empirical 不依理論，仍產出有效結果"""
    cfg = SimulationConfig(
        modulation="16QAM",
        channel="Rayleigh",
        snr_db_range=(0, 8),
        snr_db_step=4,
        num_trials=30,
        bits_per_simulation=500,
        random_seed=42,
    )
    engine = MonteCarloEngine(cfg)
    r = engine.run_adaptive_empirical(
        snr_points=[0, 4, 8],
        base_trials=20,
        max_trials_per_point=200,
        target_se=5e-3,
        min_trials=25,
    )
    assert len(r["ber"]) == 3
    assert len(r["num_trials_per_point"]) == 3
    assert all(r["ber"][i] >= 0 and r["ber"][i] <= 1 for i in range(3))
