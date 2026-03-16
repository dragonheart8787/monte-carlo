"""
報表產生器

自動整理：
- 單次實驗摘要
- 多實驗比較摘要
- 圖表清單
- 主要觀察結論模板
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import normalize_result
from .diagnostics import diagnose_results, format_diagnostics
from .theory import get_theoretical_ber


def build_single_report(
    results: Dict[str, Any],
    experiment_id: str = "",
    output_path: Optional[str] = None,
) -> str:
    """單次實驗摘要報表"""
    r = normalize_result(results)
    cfg = r.get("config", r.get("config_snapshot", {}))
    diag = diagnose_results(r, cfg)

    lines = [
        "=" * 60,
        "蒙地卡羅通訊模擬 - 單次實驗報告",
        "=" * 60,
        f"實驗 ID: {experiment_id or r.get('exp_id', 'N/A')}",
        f"時間: {r.get('timestamp', datetime.now().isoformat())}",
        "",
        "【配置】",
    ]
    for k, v in cfg.items():
        lines.append(f"  {k}: {v}")

    lines.extend([
        "",
        "【BER vs SNR】",
        "SNR(dB) | BER(模擬) | CI_low | CI_high | 理論值 | 相對誤差",
        "-" * 60,
    ])

    snr = r.get("snr_db", [])
    ber = r.get("ber", [])
    ci_low = r.get("ber_ci_low", r.get("confidence_interval_low", []))
    ci_high = r.get("ber_ci_high", r.get("confidence_interval_high", []))
    theory = r.get("theory_ber", r.get("theoretical_ber", []))
    rel_err = r.get("relative_error", [])

    for i in range(len(snr)):
        t_str = f"{theory[i]:.2e}" if i < len(theory) and theory[i] is not None else "N/A"
        re_str = f"{rel_err[i]:.1%}" if i < len(rel_err) and rel_err[i] is not None else "N/A"
        cl = ci_low[i] if i < len(ci_low) else 0
        ch = ci_high[i] if i < len(ci_high) else 0
        b = ber[i] if i < len(ber) else 0
        lines.append(f"  {snr[i]:6.1f} | {b:.2e} | {cl:.2e} | {ch:.2e} | {t_str} | {re_str}")

    if diag:
        lines.extend(["", format_diagnostics(diag)])

    lines.append("")
    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")

    return report


def build_comparison_report(
    records: List[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """多實驗比較摘要"""
    labels = labels or [f"實驗{i+1}" for i in range(len(records))]

    lines = [
        "=" * 70,
        "蒙地卡羅通訊模擬 - 多實驗比較報告",
        "=" * 70,
        f"比較項目數: {len(records)}",
        "",
        "【摘要表】",
        "標籤 | 調變 | 通道 | BER@中點 | 理論值 | 相對誤差",
        "-" * 70,
    ]

    for rec, lbl in zip(records, labels):
        cfg = rec.get("config", rec.get("config_snapshot", {}))
        res = rec.get("results", rec)
        mid = len(res.get("snr_db", [])) // 2
        ber = res.get("ber", [0])[mid] if res.get("ber") else 0
        theory = res.get("theory_ber", res.get("theoretical_ber", []))
        theory_mid = theory[mid] if mid < len(theory) else None
        rel_err = res.get("relative_error", [])
        re_mid = rel_err[mid] if mid < len(rel_err) else None

        t_str = f"{theory_mid:.2e}" if theory_mid is not None else "N/A"
        re_str = f"{re_mid:.1%}" if re_mid is not None else "N/A"
        lines.append(
            f"{lbl[:12]:12} | {cfg.get('modulation',''):6} | "
            f"{cfg.get('channel',''):8} | {ber:.2e} | {t_str} | {re_str}"
        )

    lines.append("")
    lines.append("【主要觀察】")
    lines.append("（請依實驗目的補充結論）")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")

    return report


def get_figure_checklist(exp_dir: Path) -> List[str]:
    """取得實驗目錄下應有的圖表清單"""
    figures_dir = exp_dir / "figures"
    if not figures_dir.exists():
        return []
    return [str(p.name) for p in figures_dir.glob("*") if p.suffix in {".png", ".pdf", ".jpg"}]
