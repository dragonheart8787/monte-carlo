"""
實驗管理層

負責：
- 批次執行多組 modulation / channel / SNR 組合
- 為每組實驗建立唯一 ID
- 儲存 config snapshot、metadata、results
- 管理 artifacts/、logs/、figures/
- 產出 summary table
"""

import json
import hashlib
import subprocess
import sys
import time
import uuid

import numpy as np
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import SimulationConfig
from .monte_carlo_engine import MonteCarloEngine
from .theory import get_theoretical_ber, relative_error, absolute_error
from .schemas import normalize_result, SCHEMA_VERSION, SYSTEM_VERSION
from .diagnostics import diagnose_results, format_diagnostics
from .report_builder import build_single_report


@dataclass
class ExperimentRecord:
    """單一實驗記錄"""

    experiment_id: str
    config_snapshot: Dict[str, Any]
    results: Dict[str, Any]
    timestamp: str
    output_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _get_git_commit() -> Optional[str]:
    """取得當前 git commit（若在 git repo 內）"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


class ExperimentManager:
    """實驗編排與管理"""

    def __init__(
        self,
        output_root: str = "experiments",
        prefix: str = "exp",
    ):
        self.output_root = Path(output_root)
        self.prefix = prefix
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._records: List[ExperimentRecord] = []

    def _generate_id(self, config: SimulationConfig) -> str:
        """產生實驗唯一 ID"""
        config_str = json.dumps(
            {
                "mod": config.modulation,
                "ch": config.channel,
                "snr": str(config.snr_db_range),
                "seed": config.random_seed,
                "t": datetime.now().isoformat()[:19],
            },
            sort_keys=True,
        )
        h = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        return f"{self.prefix}_{h}_{uuid.uuid4().hex[:4]}"

    def _create_exp_structure(self, exp_dir: Path) -> None:
        """建立實驗目錄結構"""
        (exp_dir / "artifacts").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "figures").mkdir(exist_ok=True)

    def run_single(
        self,
        config: SimulationConfig,
        experiment_id: Optional[str] = None,
        save_results: bool = True,
        save_figures: bool = True,
        description: str = "",
    ) -> ExperimentRecord:
        """執行單一實驗並記錄"""
        eid = experiment_id or self._generate_id(config)
        exp_dir = self.output_root / eid
        exp_dir.mkdir(parents=True, exist_ok=True)
        self._create_exp_structure(exp_dir)

        # 儲存 config
        config_path = exp_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)

        # 執行
        t0 = time.perf_counter()
        engine = MonteCarloEngine(config)
        results = engine.run()
        runtime_sec = time.perf_counter() - t0

        # 加入理論值與誤差
        theory = get_theoretical_ber(
            config.modulation,
            config.channel,
            results["snr_db"],
            config.rician_k,
        )
        if theory is not None:
            results["theory_ber"] = list(np.asarray(theory).flatten())
            rel_err = []
            abs_err = []
            for s, t in zip(results["ber"], results["theory_ber"]):
                rel_err.append(relative_error(s, t))
                abs_err.append(absolute_error(s, t))
            results["relative_error"] = rel_err
            results["absolute_error"] = abs_err

        results["exp_id"] = eid
        results["timestamp"] = datetime.now().isoformat()
        results["runtime_sec"] = runtime_sec
        results["schema_version"] = SCHEMA_VERSION
        results["system_version"] = SYSTEM_VERSION
        results["config"] = {**results.get("config", {}), **asdict(config)}

        # 診斷
        diag = diagnose_results(results, results.get("config", asdict(config)))
        # diagnostics 含 non-JSON 物件，不寫入 results
        if diag:
            log_path = exp_dir / "logs" / "diagnostics.txt"
            log_path.write_text(format_diagnostics(diag), encoding="utf-8")

        theory_avail = config.theory_available
        if theory_avail is None:
            theory_avail = get_theoretical_ber(
                config.modulation, config.channel, [0], config.rician_k
            ) is not None

        # metadata.json
        metadata = {
            "exp_id": eid,
            "created_at": datetime.now().isoformat(),
            "schema_version": SCHEMA_VERSION,
            "system_version": SYSTEM_VERSION,
            "git_commit": _get_git_commit(),
            "python_version": sys.version.split()[0],
            "random_seed": config.random_seed,
            "status": "completed",
            "tags": [],
            "description": description,
            "runtime_sec": runtime_sec,
            "warnings": [d["message"] for d in diag if d["level"] == "warning"],
            "experiment_type": getattr(config, "experiment_type", "exploratory"),
            "experiment_rigor": getattr(config, "experiment_rigor", "baseline"),
            "theory_available": theory_avail,
        }
        metadata_path = exp_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if save_results:
            results_copy = {k: v for k, v in results.items() if k != "diagnostics"}
            results_serializable = self._to_serializable(results_copy)
            results_path = exp_dir / "results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        if save_figures:
            from .visualizer import Visualizer
            from .dashboard import build_dashboard
            viz = Visualizer()
            fig_path = exp_dir / "figures" / "ber_vs_snr.png"
            viz.plot_ber_vs_snr(
                results,
                theoretical=True,
                show_ci=True,
                save_path=str(fig_path),
            )
            dashboard_path = exp_dir / "figures" / "dashboard.png"
            build_dashboard(results, str(dashboard_path))

        # 報表
        report_path = exp_dir / "report.txt"
        build_single_report(results, eid, str(report_path))

        record = ExperimentRecord(
            experiment_id=eid,
            config_snapshot=asdict(config),
            results=results,
            timestamp=metadata["created_at"],
            output_dir=str(exp_dir),
            metadata={"config_path": str(config_path), "metadata_path": str(metadata_path)},
        )
        self._records.append(record)
        return record

    def run_sweep(
        self,
        modulations: List[str],
        channels: List[str],
        base_config: Optional[SimulationConfig] = None,
        snr_range: Optional[Tuple[float, float]] = None,
        snr_step: float = 2.0,
    ) -> List[ExperimentRecord]:
        """批次掃描多組 modulation × channel 組合"""
        base = base_config or SimulationConfig()
        if snr_range:
            base.snr_db_range = snr_range
            base.snr_db_step = snr_step

        records = []
        for mod in modulations:
            for ch in channels:
                cfg = SimulationConfig(
                    modulation=mod,
                    channel=ch,
                    snr_db_range=base.snr_db_range,
                    snr_db_step=base.snr_db_step,
                    bits_per_simulation=base.bits_per_simulation,
                    num_trials=base.num_trials,
                    random_seed=base.random_seed,
                    rician_k=base.rician_k,
                )
                record = self.run_single(cfg, save_results=True)
                records.append(record)

        return records

    def run_custom_sweep(
        self,
        configs: List[SimulationConfig],
    ) -> List[ExperimentRecord]:
        """自訂多組 config 批次執行"""
        return [self.run_single(cfg) for cfg in configs]

    def generate_summary_table(
        self,
        records: Optional[List[ExperimentRecord]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """產出 summary table"""
        recs = records or self._records
        if not recs:
            return ""

        lines = [
            "=" * 90,
            "實驗摘要表",
            "=" * 90,
            "實驗 ID | 調變 | 通道 | 類型 | 嚴謹度 | 理論 | SNR範圍 | 試驗數 | BER@中點 | 理論值 | 相對誤差",
            "-" * 90,
        ]

        for r in recs:
            cfg = r.config_snapshot
            res = r.results
            mid = len(res.get("snr_db", [])) // 2
            ber_mid = res.get("ber", [0])[mid] if res.get("ber") else 0
            theory_mid = res.get("theory_ber", [None])[mid] if res.get("theory_ber") else None
            rel_err = res.get("relative_error", [None])[mid] if res.get("relative_error") else None

            theory_str = f"{theory_mid:.2e}" if theory_mid is not None else "N/A"
            rel_str = f"{rel_err:.2%}" if rel_err is not None else "N/A"
            exp_type = cfg.get("experiment_type", "N/A")
            exp_rigor = cfg.get("experiment_rigor", "N/A")
            theory_av = cfg.get("theory_available")
            theory_av_str = "Y" if theory_av else "N" if theory_av is False else "?"

            lines.append(
                f"{r.experiment_id[:16]:16} | {cfg.get('modulation',''):6} | "
                f"{cfg.get('channel',''):8} | {str(exp_type)[:10]:10} | "
                f"{str(exp_rigor)[:8]:8} | {theory_av_str} | "
                f"{str(cfg.get('snr_db_range','')):10} | {cfg.get('num_trials',0):6} | "
                f"{ber_mid:.2e} | {theory_str} | {rel_str}"
            )

        lines.append("")
        summary = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(summary, encoding="utf-8")

        return summary

    def get_record(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """依 ID 取得實驗記錄"""
        for r in self._records:
            if r.experiment_id == experiment_id:
                return r
        return None

    def load_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """從磁碟載入已儲存的實驗"""
        exp_dir = self.output_root / experiment_id
        if not exp_dir.is_dir():
            return None

        config_path = exp_dir / "config.json"
        results_path = exp_dir / "results.json"
        metadata_path = exp_dir / "metadata.json"

        if not config_path.exists() or not results_path.exists():
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            config_snapshot = json.load(f)

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return ExperimentRecord(
            experiment_id=experiment_id,
            config_snapshot=config_snapshot,
            results=results,
            timestamp=metadata.get("created_at", ""),
            output_dir=str(exp_dir),
            metadata=metadata,
        )

    def _to_serializable(self, obj: Any) -> Any:
        """將 numpy 等轉為 JSON 可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        return obj
