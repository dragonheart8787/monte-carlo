"""
診斷與警告系統

提供高 SNR 區域樣本不足、relative error 過大、
理論值不可用等情境的診斷提示。
"""

from typing import Dict, List, Optional, Any


def diagnose_results(
    results: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    對實驗結果進行診斷，回傳警告列表。
    每項為 {"level": "warning"|"info", "code": str, "message": str}
    """
    diag = []
    cfg = config or results.get("config", {})

    ber = results.get("ber", [])
    snr_db = results.get("snr_db", [])
    theory = results.get("theory_ber", results.get("theoretical_ber"))
    rel_err = results.get("relative_error", [])
    raw = results.get("raw_results", [])

    # 高 SNR 區域 errors=0，樣本數可能不足
    if raw and ber:
        for i, r in enumerate(raw):
            if i < len(ber) and ber[i] == 0 and i < len(snr_db) and snr_db[i] >= 8:
                diag.append({
                    "level": "warning",
                    "code": "LOW_SAMPLES_HIGH_SNR",
                    "message": f"SNR={snr_db[i]:.1f}dB 時 BER=0，樣本數可能不足，建議增加 num_trials 或 bits_per_simulation",
                })
                break

    # relative error 過大
    if rel_err:
        max_rel = max(rel_err)
        if max_rel > 0.5:
            diag.append({
                "level": "warning",
                "code": "HIGH_RELATIVE_ERROR",
                "message": f"最大相對誤差 {max_rel:.1%}，模擬可能未收斂",
            })

    # 理論值不可用
    if theory is None and cfg:
        mod, ch = cfg.get("modulation", ""), cfg.get("channel", "")
        diag.append({
            "level": "info",
            "code": "NO_THEORY",
            "message": f"{mod}+{ch} 無理論公式，比較僅供模擬間參考",
        })

    # block_size 不合理
    if cfg.get("fading_mode") == "block":
        blk = cfg.get("block_size", 0)
        bits = cfg.get("bits_per_simulation", 0)
        if blk > bits // 2:
            diag.append({
                "level": "warning",
                "code": "BLOCK_SIZE_LARGE",
                "message": f"block_size ({blk}) 較大，block 數量可能不足",
            })

    # 低 trial 數
    num_trials = cfg.get("num_trials", 0)
    if num_trials and num_trials < 30:
        diag.append({
            "level": "info",
            "code": "LOW_TRIALS",
            "message": f"num_trials={num_trials}，信賴區間可能較寬",
        })

    return diag


def format_diagnostics(diag: List[Dict[str, str]]) -> str:
    """將診斷結果格式化為可讀文字"""
    if not diag:
        return ""
    lines = ["診斷結果："]
    for d in diag:
        prefix = "[WARN]" if d["level"] == "warning" else "[INFO]"
        lines.append(f"  {prefix} {d['message']}")
    return "\n".join(lines)
