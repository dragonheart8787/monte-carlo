# 限制與已知假設

本文件明確列出系統的限制與假設，避免誤用或過度解讀。

---

## 1. Uncoded Link Only

- **現狀**：僅模擬未編碼鏈路（uncoded link）
- **不支援**：Hamming、Turbo、LDPC、Convolutional 等編碼
- **影響**：BER/PER 為 raw channel 表現，非實際系統（通常會加 FEC）

---

## 2. Coherent Detection Assumptions

- **假設**：接收端具備載波同步（carrier recovery）
- **理論公式**：BPSK/QPSK/M-PSK/16-QAM 理論值皆假設 coherent detection
- **不適用**：Non-coherent (e.g. DPSK)、differential detection 需另行建模

---

## 3. Theory Support Coverage

| 調變 | AWGN | Rayleigh | Rician |
|------|------|----------|--------|
| BPSK | ✓ closed-form | ✓ closed-form | ✓ 近似 |
| QPSK | ✓ closed-form | ✓ closed-form | ✗ |
| 8-PSK | ✓ 近似 | ✗ | ✗ |
| 16-QAM | ✓ 近似 | ✗ | ✗ |

- **無理論時**：`theory_available=False`，比較僅供模擬間參考
- **Rician**：使用 K 因子插值近似，非 exact closed-form

---

## 4. CI Stopping 不適用的極端區間

- **BER ≈ 0**：`target_ci_width_relative` 分母趨近 0，不參與判斷
- **errors = 0**：Wilson score 仍可算 CI，但寬度可能很大
- **極低 trial 數**：CI 寬度不可靠，建議 `min_trials >= 20`
- **高變異區**：可能需大量樣本才達標，注意 `max_trials` 上限

---

## 5. Adaptive Strategy 限制

**Theory-guided (`run_adaptive`)**：
- 依賴 `get_theoretical_ber()` 回傳值
- 理論不可用時退化为固定 `base_trials`
- 理論與模擬 channel 假設不一致時，alloc 可能不優

**Empirical (`run_adaptive_empirical`)**：
- 以 SE 為目標，不依理論
- 低 BER 區可能需較多樣本才能達 `target_se`
- 無理論時為唯一選擇

---

## 6. Impairment Pipeline 固定順序

`CombinedImpairment` 順序固定，不可變更：

1. phase_offset
2. frequency_offset
3. amplitude（若未來擴充）
4. other

**不同順序數學上不等價**，結果無法與他案比較。

---

## 7. 其他限制

- **單一載波**：無 OFDM、無多徑延遲建模
- **無 IQ imbalance**：未建模
- **無 timing offset**：符號同步假設完美
- **Block fading**：block 內增益恆定，block 間獨立
