"""
蒙地卡羅通訊效能評估系統

以蒙地卡羅方法為核心，在不同雜訊、衰落、調變與接收條件下，
自動模擬並分析數位通訊鏈路的錯誤率、收斂特性與系統可靠性。
"""

from .config import SimulationConfig
from .data_generator import DataGenerator
from .modulation import Modulator
from .channel import Channel
from .channel_state import ChannelState
from .channel_processes import (
    FadingProcess,
    AWGNFading,
    RayleighFading,
    RicianFading,
    AdditiveNoiseProcess,
    AWGNNoise,
    BurstAWGNNoise,
    ImpairmentProcess,
)
from .receiver import Receiver
from .performance import PerformanceEvaluator
from .monte_carlo_engine import MonteCarloEngine
from .visualizer import Visualizer
from .theory import get_theoretical_ber, relative_error, absolute_error
from .experiment_manager import ExperimentManager, ExperimentRecord
from .validators import validate_config, ValidationError
from .schemas import normalize_result, validate_result_schema
from .diagnostics import diagnose_results, format_diagnostics
from .report_builder import build_single_report, build_comparison_report
from .benchmark_suite import run_benchmark_suite, BENCHMARK_CASES
from .dashboard import build_dashboard

__all__ = [
    "SimulationConfig",
    "DataGenerator",
    "Modulator",
    "Channel",
    "Receiver",
    "PerformanceEvaluator",
    "MonteCarloEngine",
    "Visualizer",
    "get_theoretical_ber",
    "relative_error",
    "absolute_error",
    "ExperimentManager",
    "ExperimentRecord",
    "validate_config",
    "ValidationError",
    "normalize_result",
    "validate_result_schema",
    "diagnose_results",
    "format_diagnostics",
    "build_single_report",
    "build_comparison_report",
    "run_benchmark_suite",
    "BENCHMARK_CASES",
    "build_dashboard",
    "FadingProcess",
    "AWGNFading",
    "RayleighFading",
    "RicianFading",
    "AdditiveNoiseProcess",
    "AWGNNoise",
    "BurstAWGNNoise",
    "ImpairmentProcess",
    "ChannelState",
]
