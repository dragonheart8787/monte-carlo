# CI-based Stopping 統計定義

## 1. 信賴區間方法

本系統使用 **Wilson score interval** 估計 BER 信賴區間。

- 原因：二項比例在小樣本或極端比例時，normal approximation 不準
- Wilson score 在 p 接近 0 或 1 時仍較穩定
- 公式：見 `performance.py` 之 `_ci_for_proportion(p, n)`

## 2. Relative Width 定義

```
relative_width = (CI_high - CI_low) / denominator
```

**分母 (denominator)**：
- 使用估計 mean（即 BER 點估計）
- 當 mean < 1e-10 時，視為 0，relative_width 不適用，僅用 absolute width 判斷

## 3. BER 接近 0 時

- Wilson score 在 errors=0 時仍可計算（不會退化為 [0,0]）
- 當 mean_ber < 1e-10：`target_ci_width_relative` 不參與停止判斷
- 此時僅依 `target_ci_width`（絕對寬度）決定是否停止

## 4. 停止條件

- `target_ci_width`：絕對寬度 (upper - lower) ≤ 門檻 → 停止
- `target_ci_width_relative`：相對寬度 ≤ 門檻 且 mean > 1e-10 → 停止
- 兩者為 OR 關係：任一達標即停止

## 5. 與 Clopper-Pearson 的差異

- Clopper-Pearson 為 exact interval，較保守
- Wilson 為近似，但計算簡單、實務常用
- 若需 exact，可於未來擴充
