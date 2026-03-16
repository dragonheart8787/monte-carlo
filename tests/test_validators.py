"""
驗證器測試
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_comm_system.config import SimulationConfig
from mc_comm_system.validators import validate_config, ValidationError


def test_valid_config_passes():
    """合法 config 應通過驗證"""
    cfg = SimulationConfig(
        modulation="BPSK",
        channel="AWGN",
        block_size=50,
        burst_noise_prob=0.0,
        random_seed=42,
    )
    warnings = validate_config(cfg)
    assert isinstance(warnings, list)


def test_invalid_modulation_raises():
    """非法 modulation 應拋出 ValidationError"""
    cfg = SimulationConfig(modulation="INVALID", channel="AWGN")
    try:
        validate_config(cfg)
        assert False, "應拋出 ValidationError"
    except ValidationError:
        pass


def test_invalid_block_size_raises():
    """block_size <= 0 應拋出"""
    cfg = SimulationConfig(block_size=0, channel="AWGN")
    try:
        validate_config(cfg)
        assert False, "應拋出 ValidationError"
    except ValidationError:
        pass


def test_burst_prob_out_of_range_raises():
    """burst_noise_prob 超出 [0,1] 應拋出"""
    cfg = SimulationConfig(burst_noise_prob=1.5, channel="AWGN")
    try:
        validate_config(cfg)
        assert False, "應拋出 ValidationError"
    except ValidationError:
        pass
