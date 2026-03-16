"""
Channel State 契約

明確定義 channel_state 結構，避免各模組對 key 名稱、欄位理解不一致。
Compensator / Detector 皆應依此契約讀寫。
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ChannelState:
    """
    通道狀態契約（供 Compensator 使用）

    欄位說明：
    - fading_coefficients: 複數 fading 增益 h[n]，若為 None 表示未知
    - phase_offset_rad: 相位偏移 (rad)
    - freq_offset_norm: 正規化頻率偏移
    - impairment_metadata: 其他擾動描述（如 IQ imbalance 等）
    - perfect_csi_available: 是否具備完美通道估計（理想補償情境）
    """

    fading_coefficients: Optional[List[complex]] = None
    phase_offset_rad: Optional[float] = None
    freq_offset_norm: Optional[float] = None
    impairment_metadata: Optional[dict] = None
    perfect_csi_available: bool = False

    def to_dict(self) -> dict:
        """轉為 dict（供 JSON 或相容舊介面）"""
        return {
            "fading_coefficients": self.fading_coefficients,
            "phase_offset_rad": self.phase_offset_rad,
            "freq_offset_norm": self.freq_offset_norm,
            "impairment_metadata": self.impairment_metadata,
            "perfect_csi_available": self.perfect_csi_available,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelState":
        """從 dict 建立"""
        return cls(
            fading_coefficients=d.get("fading_coefficients"),
            phase_offset_rad=d.get("phase_offset_rad"),
            freq_offset_norm=d.get("freq_offset_norm"),
            impairment_metadata=d.get("impairment_metadata"),
            perfect_csi_available=d.get("perfect_csi_available", False),
        )
