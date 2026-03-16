"""
執行官方基準實驗組合

每次改版可執行此腳本，確保回歸通過。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mc_comm_system.benchmark_suite import run_benchmark_suite

if __name__ == "__main__":
    result = run_benchmark_suite(
        output_root="benchmark_results",
        save_figures=True,
    )
    print(result["summary_table"])
    print("\nPassed:", result["passed"])
    print("Failed:", result["failed"])
    print("All passed:", result["all_passed"])
    print("Summary dashboard:", result.get("summary_dashboard_path", "N/A"))
    sys.exit(0 if result["all_passed"] else 1)
