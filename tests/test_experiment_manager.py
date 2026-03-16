"""
實驗管理測試

- run_single() 會產出完整資料夾
- load_experiment() 能回讀一致資料
- run_sweep() 筆數正確
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_comm_system import SimulationConfig, ExperimentManager


def test_run_single_produces_folder():
    """run_single 產出 config, results, metadata, artifacts, logs, figures"""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = ExperimentManager(output_root=tmp, prefix="t")
        cfg = SimulationConfig(
            modulation="BPSK",
            channel="AWGN",
            snr_db_range=(0, 4),
            snr_db_step=2,
            num_trials=3,
            bits_per_simulation=500,
            random_seed=42,
        )
        rec = mgr.run_single(cfg, experiment_id="test_exp", save_figures=False)

        exp_dir = Path(tmp) / "test_exp"
        assert exp_dir.exists()
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "results.json").exists()
        assert (exp_dir / "metadata.json").exists()
        assert (exp_dir / "artifacts").is_dir()
        assert (exp_dir / "logs").is_dir()
        assert (exp_dir / "figures").is_dir()
        assert (exp_dir / "report.txt").exists()

        with open(exp_dir / "results.json", encoding="utf-8") as f:
            res = json.load(f)
        assert "snr_db" in res
        assert "ber" in res
        assert len(res["ber"]) == len(res["snr_db"])


def test_load_experiment_consistent():
    """load_experiment 能回讀與 run_single 一致的資料"""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = ExperimentManager(output_root=tmp, prefix="t")
        cfg = SimulationConfig(
            modulation="BPSK",
            channel="AWGN",
            snr_db_range=(0, 2),
            snr_db_step=2,
            num_trials=2,
            bits_per_simulation=200,
            random_seed=42,
        )
        rec1 = mgr.run_single(cfg, experiment_id="load_test", save_figures=False)

        loaded = mgr.load_experiment("load_test")
        assert loaded is not None
        assert loaded.experiment_id == "load_test"
        assert loaded.results["snr_db"] == rec1.results["snr_db"]
        assert loaded.config_snapshot["modulation"] == "BPSK"


def test_run_sweep_count():
    """run_sweep 產出正確筆數"""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = ExperimentManager(output_root=tmp, prefix="t")
        records = mgr.run_sweep(
            modulations=["BPSK", "QPSK"],
            channels=["AWGN", "Rayleigh"],
            base_config=SimulationConfig(
                snr_db_range=(0, 2),
                snr_db_step=2,
                num_trials=2,
                bits_per_simulation=200,
                random_seed=42,
            ),
        )
        assert len(records) == 4  # 2 mod × 2 ch
