"""
結果格式契約 (Result Schema)

定義實驗結果的固定欄位，確保 visualizer、report、load_experiment 等模組
一致使用相同結構。

所有模組皆應依此契約輸出/讀取結果。
"""

from typing import Any, Dict, List, Optional, TypedDict

# Schema 版本：修改欄位時須遞增，舊結果可依此判斷相容性
SCHEMA_VERSION = "1.0"
SYSTEM_VERSION = "1.0"


class ExperimentResultSchema(TypedDict, total=False):
    """實驗結果固定欄位"""

    # 識別
    exp_id: str
    timestamp: str

    # 快照
    config: Dict[str, Any]
    config_snapshot: Dict[str, Any]

    # SNR 掃描
    snr_db: List[float]

    # 效能指標
    ber: List[float]
    ser: List[float]
    per: List[float]

    # 信賴區間
    ber_ci_low: List[float]
    ber_ci_high: List[float]
    confidence_interval_low: List[float]
    confidence_interval_high: List[float]

    # 誤差
    ber_se: List[float]
    standard_error: List[float]

    # 理論與比較
    theory_ber: List[float]
    theoretical_ber: List[float]
    absolute_error: List[float]
    relative_error: List[float]

    # 樣本統計
    num_bits: int
    num_symbols: int
    num_packets: int
    num_errors: List[int]
    num_trials: List[int]
    total_bits: int

    # 原始
    raw_results: List[Dict[str, Any]]

    # 執行
    runtime_sec: float
    notes: str

    # 版本（供 load/compare 判斷相容性）
    schema_version: str
    system_version: str


# 標準欄位名稱對照（供相容性用）
REQUIRED_FIELDS = ["snr_db", "ber", "config"]
OPTIONAL_FIELDS = [
    "ber_ci_low",
    "ber_ci_high",
    "ber_se",
    "theory_ber",
    "absolute_error",
    "relative_error",
    "raw_results",
    "runtime_sec",
]


def normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    將結果正規化為標準 schema。
    相容舊欄位名稱（如 theory_ber -> theoretical_ber 別名）。
    """
    out = dict(result)
    if "theory_ber" in out and "theoretical_ber" not in out:
        out["theoretical_ber"] = out["theory_ber"]
    if "ber_ci_low" in out and "confidence_interval_low" not in out:
        out["confidence_interval_low"] = out["ber_ci_low"]
    if "ber_ci_high" in out and "confidence_interval_high" not in out:
        out["confidence_interval_high"] = out["ber_ci_high"]
    if "ber_se" in out and "standard_error" not in out:
        out["standard_error"] = out["ber_se"]
    return out


def validate_result_schema(result: Dict[str, Any]) -> List[str]:
    """
    檢查結果是否符合 schema，回傳缺失的 required 欄位。
    """
    missing = []
    for f in REQUIRED_FIELDS:
        if f not in result or result[f] is None:
            missing.append(f)
    return missing
