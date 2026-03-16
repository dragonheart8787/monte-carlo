"""
參數驗證層

驗證 config 與實驗參數的合理性，防止錯參數導致系統異常。
"""

from typing import List, Optional, Tuple

from .config import SimulationConfig


VALID_MODULATION = {"BPSK", "QPSK", "8PSK", "16QAM"}
VALID_CHANNEL = {"AWGN", "RAYLEIGH", "RICIAN"}
VALID_DECISION = {"hard", "soft"}
VALID_CODING = {None, "hamming", "repetition"}
VALID_FADING_MODE = {"fast", "block"}


class ValidationError(ValueError):
    """驗證失敗"""
    pass


def validate_modulation(modulation: str) -> None:
    if modulation.upper() not in VALID_MODULATION:
        raise ValidationError(f"modulation 必須為 {VALID_MODULATION}")


def validate_channel(channel: str) -> None:
    if channel.upper() not in VALID_CHANNEL:
        raise ValidationError(f"channel 必須為 {VALID_CHANNEL}")


def validate_block_size(block_size: int, fading_mode: str) -> None:
    if block_size <= 0:
        raise ValidationError("block_size 必須 > 0")
    if fading_mode.lower() == "block" and block_size <= 0:
        raise ValidationError("block fading 時 block_size 必須 > 0")


def validate_burst_noise(burst_noise_prob: float, burst_noise_ratio: float) -> None:
    if not 0 <= burst_noise_prob <= 1:
        raise ValidationError("burst_noise_prob 必須在 [0, 1]")
    if burst_noise_ratio < 1:
        raise ValidationError("burst_noise_ratio 必須 >= 1")


def validate_freq_offset(freq_offset_norm: float) -> None:
    if abs(freq_offset_norm) > 0.5:
        raise ValidationError(
            "freq_offset_norm 建議在 [-0.5, 0.5]，過大可能導致嚴重失真"
        )


def validate_rician_k(channel: str, rician_k: float) -> None:
    if channel.upper() == "RICIAN" and rician_k < 0:
        raise ValidationError("Rician 通道時 rician_k 必須 >= 0")


def validate_snr_range(snr_db_range: Tuple[float, float], snr_db_step: float) -> None:
    low, high = snr_db_range
    if low > high:
        raise ValidationError("snr_db_range 低限不可大於高限")
    if snr_db_step <= 0:
        raise ValidationError("snr_db_step 必須 > 0")


def validate_sample_params(
    bits_per_simulation: int,
    num_trials: int,
    packet_length: Optional[int],
    block_size: int,
    fading_mode: str,
) -> None:
    if bits_per_simulation <= 0:
        raise ValidationError("bits_per_simulation 必須 > 0")
    if num_trials <= 0:
        raise ValidationError("num_trials 必須 > 0")
    if packet_length is not None and packet_length <= 0:
        raise ValidationError("packet_length 必須 > 0（若指定）")
    if fading_mode.lower() == "block" and block_size > bits_per_simulation:
        raise ValidationError(
            f"block_size ({block_size}) 不應大於 bits_per_simulation ({bits_per_simulation})"
        )


def validate_config(config: SimulationConfig) -> List[str]:
    """
    完整驗證 config，回傳 warnings 列表。
    若驗證失敗則拋出 ValidationError。
    """
    warnings = []

    validate_modulation(config.modulation)
    validate_channel(config.channel)
    validate_block_size(config.block_size, config.fading_mode)
    validate_burst_noise(config.burst_noise_prob, config.burst_noise_ratio)
    validate_freq_offset(config.freq_offset_norm)
    validate_rician_k(config.channel, config.rician_k)
    validate_snr_range(config.snr_db_range, config.snr_db_step)

    if config.decision_type not in VALID_DECISION:
        raise ValidationError(f"decision_type 必須為 {VALID_DECISION}")
    if config.coding not in VALID_CODING:
        raise ValidationError(f"coding 必須為 {VALID_CODING}")
    if config.fading_mode.lower() not in VALID_FADING_MODE:
        raise ValidationError(f"fading_mode 必須為 {VALID_FADING_MODE}")

    validate_sample_params(
        config.bits_per_simulation,
        config.num_trials,
        config.packet_length,
        config.block_size,
        config.fading_mode,
    )

    # 非致命 warnings
    if config.block_size > config.bits_per_simulation // 2:
        warnings.append(
            f"block_size ({config.block_size}) 較大，可能僅有少量 block"
        )
    if config.burst_noise_prob > 0.1:
        warnings.append("burst_noise_prob 較高，可能顯著影響 BER")
    if config.num_trials < 30:
        warnings.append("num_trials < 30，信賴區間可能較寬")

    return warnings
