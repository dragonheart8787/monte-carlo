"""
蒙地卡羅引擎進階測試

- CI-based stopping 會產出 ci_stopped 標記
- Adaptive allocation 會依 SNR 分配不同樣本數
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_comm_system import SimulationConfig, MonteCarloEngine


def test_ci_stopping_produces_result():
    """run_with_ci_stopping 產出有效結果"""
    cfg = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(6, 6),
        num_trials=100,
        bits_per_simulation=500,
        random_seed=42,
    )
    engine = MonteCarloEngine(cfg)
    r = engine.run_with_ci_stopping(
        snr_db=6.0,
        target_ci_width_relative=0.5,
        min_trials=10,
        max_trials=100,
        batch_size=5,
    )
    assert "ber_mean" in r
    assert "num_trials" in r
    assert "ci_stopped" in r
    assert r["num_trials"] >= 10


def test_adaptive_allocation():
    """run_adaptive 產出不同 SNR 的結果，且低 BER 區可能用更多 trials"""
    cfg = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        snr_db_range=(0, 8),
        snr_db_step=4,
        num_trials=30,
        bits_per_simulation=500,
        random_seed=42,
    )
    engine = MonteCarloEngine(cfg)
    r = engine.run_adaptive(
        snr_points=[0, 6],
        base_trials=15,
        max_trials_per_point=200,
        target_relative_error=0.5,
        use_theory=True,
    )
    assert len(r["snr_db"]) == 2
    assert len(r["ber"]) == 2
    assert "num_trials_per_point" in r
    assert len(r["num_trials_per_point"]) == 2
