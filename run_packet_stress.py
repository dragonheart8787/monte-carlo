"""
執行 Packet-level 壓力故事線

產出 BER vs PER、Block fading 等完整故事與 dashboard 圖。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mc_comm_system.packet_stress import run_packet_stress_suite

if __name__ == "__main__":
    out_dir = "packet_stress_results"
    suite = run_packet_stress_suite(output_dir=out_dir, save_dashboards=True)
    print("Burst noise:", suite["burst_noise_ber_vs_per"]["summary"])
    print("Block fading:", suite["block_fading_per_degradation"]["summary"])
    print(f"Results: {out_dir}/")
    print("  - packet_stress_results.json")
    print("  - burst_ber_per_dashboard.png")
    print("  - block_fading_dashboard.png")
    print("  - packet_stress_conclusions.txt")
