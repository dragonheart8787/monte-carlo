# 系統架構與資料流

## 1. 通訊鏈路與模擬流程圖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Monte Carlo Communication Simulator                       │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────────┐
  │ DataGenerator│    │  Modulator   │    │           Channel                 │
  │              │    │              │    │  ┌─────────┐ ┌─────────────────┐  │
  │ random bits  │───│ bits→symbols │───│  │ Fading  │ │  Impairment     │  │
  │  / packets   │    │ BPSK/QPSK/   │    │  │ Process │→│ (phase/freq)    │  │
  └──────────────┘    │ 8PSK/16QAM   │    │  └─────────┘ └────────┬────────┘  │
                     └──────────────┘    │           │              │           │
                                        │           ▼              ▼           │
                                        │  ┌───────────────────────────────┐  │
                                        │  │     Additive Noise Process    │  │
                                        │  │     (AWGN / Burst)            │  │
                                        │  └───────────────────────────────┘  │
                                        └──────────────────┬──────────────────┘
                                                           │
                                                           ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────────┐
  │ Performance  │    │  Receiver    │    │         received symbols          │
  │ Evaluator    │◀──│              │◀──│                                    │
  │              │    │ Compensator  │    │  (optional channel_state)        │
  │ BER/SER/PER  │    │ → Detector   │    └──────────────────────────────────┘
  │ CI / SE      │    │ → Demapper   │
  └──────────────┘    └──────────────┘
         │
         ▼
  ┌──────────────┐    ┌──────────────┐
  │ Theory       │    │ MonteCarlo   │
  │ (get_ber)    │    │ Engine       │
  │              │    │ run() /      │
  │ validation   │    │ run_adaptive│
  └──────────────┘    └──────────────┘
```

## 2. Experiment Manager / Artifacts / Reports 資料流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Experiment Manager                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  run_single() / run_sweep() / run_benchmark()
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  experiments/{exp_id}/                                                    │
  │  ├── config.json          ← SimulationConfig snapshot                    │
  │  ├── metadata.json         ← exp_id, schema_version, experiment_type,    │
  │  │                          theory_available, runtime, warnings           │
  │  ├── results.json          ← snr_db, ber, ber_ci, theory_ber, raw_results │
  │  ├── report.txt            ← build_single_report()                        │
  │  ├── artifacts/            ← 額外產物（可擴充）                           │
  │  ├── logs/                                                                │
  │  │   └── diagnostics.txt   ← 診斷摘要                                     │
  │  └── figures/                                                            │
  │      ├── ber_vs_snr.png    ← Visualizer.plot_ber_vs_snr()                 │
  │      └── dashboard.png     ← build_dashboard() 一頁總覽                    │
  └──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  benchmark_results/          (run_benchmark.py)                          │
  │  ├── bm_xxx_BPSK_AWGN_theory_validation/                                  │
  │  ├── bm_xxx_QPSK_Rayleigh_convergence/                                    │
  │  ├── bm_xxx_Burst_noise_PER_stress/                                        │
  │  ├── bm_xxx_Phase_offset_compensation/                                     │
  │  ├── benchmark_summary.txt                                                │
  │  └── benchmark_summary_dashboard.png  ← 四案總覽圖                        │
  └──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  packet_stress_results/        (run_packet_stress.py)                      │
  │  ├── packet_stress_results.json                                           │
  │  ├── burst_ber_per_dashboard.png   ← Burst BER/PER 圖                     │
  │  └── block_fading_dashboard.png    ← Fast vs Block PER 圖                 │
  └──────────────────────────────────────────────────────────────────────────┘
```
