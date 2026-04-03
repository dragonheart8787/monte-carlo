"""
模組 7：蒙地卡羅分析引擎

多層次模擬策略：
- Level 1: 標準 Monte Carlo（固定樣本數）
- Level 2: 收斂監控（running mean/variance，達門檻停止）
- Level 3: 變異數縮減（antithetic, control variate）
- Level 4: 自適應抽樣（依誤差調整樣本數）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .config import SimulationConfig
from .data_generator import DataGenerator
from .modulation import Modulator
from .channel import Channel
from .receiver import Receiver, FadingEqualizer
from .performance import PerformanceEvaluator

_FADING_CHANNELS = {"RAYLEIGH", "RICIAN"}


class MonteCarloEngine:
    """蒙地卡羅模擬引擎"""

    def __init__(self, config: SimulationConfig):
        config.validate()
        self.config = config
        self.data_gen = DataGenerator(config.random_seed)
        self.modulator = Modulator(config.modulation)
        self.channel = Channel.from_config(config)
        # 衰落通道自動使用 FadingEqualizer (zero-forcing coherent equalization)
        use_equalizer = config.channel.upper() in _FADING_CHANNELS
        self.receiver = Receiver(
            config.modulation,
            config.decision_type,
            compensator=FadingEqualizer() if use_equalizer else None,
        )
        self._use_fading_csi = use_equalizer
        self.perf = PerformanceEvaluator(config.confidence_level)

    def _run_single_trial(
        self,
        snr_db: float,
        num_bits: int,
        packet_length: Optional[int],
    ) -> Dict[str, float]:
        """單次傳輸模擬"""
        bits, pkt_starts = self.data_gen.get_bits_for_simulation(
            num_bits, packet_length
        )
        symbols = self.modulator.modulate(bits)

        # Fading 通道：傳遞 channel_state (h) 給接收器做等化
        if self._use_fading_csi:
            received, channel_state = self.channel.transmit(symbols, snr_db, return_channel_state=True)
            rx_bits = self.receiver.detect(received, channel_state)
        else:
            received = self.channel.transmit(symbols, snr_db)
            rx_bits = self.receiver.detect(received)

        result = {"ber": self.perf.compute_ber(bits, rx_bits)}

        if packet_length is not None and pkt_starts is not None:
            result["per"] = self.perf.compute_per(
                bits, rx_bits, pkt_starts, packet_length
            )

        return result

    def run_snr_point(
        self,
        snr_db: float,
        num_trials: Optional[int] = None,
        bits_per_trial: Optional[int] = None,
    ) -> Dict[str, Any]:
        """對單一 SNR 點執行蒙地卡羅模擬"""
        n_trials = num_trials or self.config.num_trials
        n_bits = bits_per_trial or self.config.bits_per_simulation
        pkt_len = self.config.packet_length

        ber_list = []
        per_list = [] if pkt_len else None

        for _ in range(n_trials):
            r = self._run_single_trial(snr_db, n_bits, pkt_len)
            ber_list.append(r["ber"])
            if per_list is not None and "per" in r:
                per_list.append(r["per"])

        agg = self.perf.aggregate_results(ber_list, per_samples=per_list)
        agg["snr_db"] = snr_db
        agg["num_trials"] = n_trials
        agg["total_bits"] = n_trials * n_bits
        return agg

    def run(
        self,
        high_snr_boost: bool = False,
        high_snr_threshold_db: float = 8.0,
        high_snr_multiplier: int = 3,
    ) -> Dict[str, Any]:
        """
        執行完整 SNR 掃描模擬。
        high_snr_boost: 若 True，SNR >= threshold 的點使用 multiplier 倍試驗數。
        """
        snr_points = self.config.get_snr_points()
        results = []
        base_trials = self.config.num_trials

        for snr_db in snr_points:
            n_trials = base_trials
            if high_snr_boost and snr_db >= high_snr_threshold_db:
                n_trials = base_trials * high_snr_multiplier
            r = self.run_snr_point(snr_db, num_trials=n_trials)
            results.append(r)

        return {
            "config": {
                "modulation": self.config.modulation,
                "channel": self.config.channel,
                "rician_k": self.config.rician_k,
                "num_trials": self.config.num_trials,
                "bits_per_simulation": self.config.bits_per_simulation,
            },
            "snr_db": [r["snr_db"] for r in results],
            "ber": [r["ber_mean"] for r in results],
            "ber_ci_low": [r["ber_ci_low"] for r in results],
            "ber_ci_high": [r["ber_ci_high"] for r in results],
            "ber_se": [r["ber_se"] for r in results],
            "raw_results": results,
        }

    def run_with_convergence_monitor(
        self,
        snr_db: float,
        target_se: float = 1e-4,
        min_trials: int = 10,
        max_trials: int = 10000,
    ) -> Dict[str, Any]:
        """
        Level 2: 收斂監控
        當標準誤差小於 target_se 或達到 max_trials 時停止。
        """
        ber_list = []
        n_bits = self.config.bits_per_simulation
        pkt_len = self.config.packet_length

        for trial in range(max_trials):
            r = self._run_single_trial(snr_db, n_bits, pkt_len)
            ber_list.append(r["ber"])

            if trial + 1 >= min_trials:
                se = self.perf.standard_error(np.mean(ber_list), len(ber_list))
                if se <= target_se:
                    break

        agg = self.perf.aggregate_results(ber_list)
        agg["snr_db"] = snr_db
        agg["num_trials"] = len(ber_list)
        agg["converged"] = len(ber_list) < max_trials
        return agg

    def run_with_ci_stopping(
        self,
        snr_db: float,
        target_ci_width: Optional[float] = None,
        target_ci_width_relative: Optional[float] = 0.1,
        min_trials: int = 20,
        max_trials: int = 50000,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """
        CI-based stopping: 依信賴區間寬度自動停止。

        - target_ci_width: 絕對寬度 (upper - lower) 門檻，達標即停
        - target_ci_width_relative: 相對寬度 (width/mean) 門檻，達標即停
        - 兩者皆達標或任一達標（依實作）即停止
        """
        ber_list = []
        n_bits = self.config.bits_per_simulation
        pkt_len = self.config.packet_length

        for trial in range(max_trials):
            r = self._run_single_trial(snr_db, n_bits, pkt_len)
            ber_list.append(r["ber"])

            if trial + 1 >= min_trials and (trial + 1) % batch_size == 0:
                mean_ber = np.mean(ber_list)
                ci_low, ci_high = self.perf._ci_for_proportion(
                    mean_ber, len(ber_list)
                )
                width = ci_high - ci_low

                stop = False
                if target_ci_width is not None and width <= target_ci_width:
                    stop = True
                if target_ci_width_relative is not None and mean_ber > 1e-10:
                    rel_width = width / mean_ber
                    if rel_width <= target_ci_width_relative:
                        stop = True
                if stop:
                    break

        agg = self.perf.aggregate_results(ber_list)
        agg["snr_db"] = snr_db
        agg["num_trials"] = len(ber_list)
        agg["ci_stopped"] = len(ber_list) < max_trials
        return agg

    def run_adaptive(
        self,
        snr_points: Optional[List[float]] = None,
        base_trials: int = 50,
        min_trials: int = 20,
        max_trials_per_point: int = 5000,
        target_relative_error: float = 0.2,
        use_theory: bool = True,
    ) -> Dict[str, Any]:
        """
        Adaptive sample allocation: 依理論 BER 與誤差動態調整樣本數。

        - 高 BER 區：較少樣本（收斂快）
        - 低 BER 區：較多樣本（需更多才能估計）
        - 若有理論值，以 relative_error 為目標調整
        """
        from .theory import get_theoretical_ber

        snr_list = snr_points or self.config.get_snr_points()
        n_bits = self.config.bits_per_simulation
        pkt_len = self.config.packet_length

        theory = None
        if use_theory:
            theory = get_theoretical_ber(
                self.config.modulation,
                self.config.channel,
                snr_list,
                self.config.rician_k,
            )

        results = []
        for i, snr_db in enumerate(snr_list):
            ber_list = []
            n_trials = base_trials
            theory_ber = float(theory[i]) if theory is not None and i < len(theory) else None

            for _ in range(max_trials_per_point):
                r = self._run_single_trial(snr_db, n_bits, pkt_len)
                ber_list.append(r["ber"])

                if len(ber_list) >= n_trials:
                    mean_ber = np.mean(ber_list)
                    rel_err = None
                    if theory_ber is not None and theory_ber > 1e-15:
                        rel_err = abs(mean_ber - theory_ber) / theory_ber

                    if rel_err is not None and rel_err <= target_relative_error:
                        break
                    if len(ber_list) >= max_trials_per_point:
                        break

                    # 未達標：增加樣本（低 BER 時需更多）
                    if theory_ber is not None and theory_ber < 1e-3:
                        n_trials = min(
                            n_trials + 100,
                            max_trials_per_point,
                        )
                    else:
                        n_trials = min(
                            n_trials + 50,
                            max_trials_per_point,
                        )

            agg = self.perf.aggregate_results(ber_list)
            agg["snr_db"] = snr_db
            agg["num_trials"] = len(ber_list)
            agg["adaptive"] = True
            results.append(agg)

        return {
            "config": {
                "modulation": self.config.modulation,
                "channel": self.config.channel,
                "rician_k": self.config.rician_k,
                "bits_per_simulation": n_bits,
            },
            "snr_db": [r["snr_db"] for r in results],
            "ber": [r["ber_mean"] for r in results],
            "ber_ci_low": [r["ber_ci_low"] for r in results],
            "ber_ci_high": [r["ber_ci_high"] for r in results],
            "ber_se": [r["ber_se"] for r in results],
            "num_trials_per_point": [r["num_trials"] for r in results],
            "raw_results": results,
        }

    def run_adaptive_empirical(
        self,
        snr_points: Optional[List[float]] = None,
        base_trials: int = 50,
        max_trials_per_point: int = 5000,
        target_se: float = 1e-4,
        min_trials: int = 30,
    ) -> Dict[str, Any]:
        """
        Purely empirical adaptive allocation：不依賴理論值。

        以 standard error 為目標，當 SE <= target_se 時停止該 SNR 點。
        適用於理論不可用或不可靠的場景。
        """
        snr_list = snr_points or self.config.get_snr_points()
        n_bits = self.config.bits_per_simulation
        pkt_len = self.config.packet_length

        results = []
        for snr_db in snr_list:
            ber_list = []
            for _ in range(max_trials_per_point):
                r = self._run_single_trial(snr_db, n_bits, pkt_len)
                ber_list.append(r["ber"])

                if len(ber_list) >= min_trials:
                    mean_ber = np.mean(ber_list)
                    se = self.perf.standard_error(mean_ber, len(ber_list))
                    if se <= target_se:
                        break

            agg = self.perf.aggregate_results(ber_list)
            agg["snr_db"] = snr_db
            agg["num_trials"] = len(ber_list)
            agg["adaptive"] = True
            agg["adaptive_mode"] = "empirical"
            results.append(agg)

        return {
            "config": {
                "modulation": self.config.modulation,
                "channel": self.config.channel,
                "rician_k": self.config.rician_k,
                "bits_per_simulation": n_bits,
            },
            "snr_db": [r["snr_db"] for r in results],
            "ber": [r["ber_mean"] for r in results],
            "ber_ci_low": [r["ber_ci_low"] for r in results],
            "ber_ci_high": [r["ber_ci_high"] for r in results],
            "ber_se": [r["ber_se"] for r in results],
            "num_trials_per_point": [r["num_trials"] for r in results],
            "raw_results": results,
        }

    def run_antithetic(self, snr_db: float, num_pairs: int) -> Dict[str, Any]:
        """
        Level 3: Antithetic Variates
        使用成對的對偶隨機數，使正負誤差互相抵消。
        注意：通道的隨機性需能配對，此處簡化為連續兩次試驗視為一對。
        """
        ber_list = []
        n_bits = self.config.bits_per_simulation
        pkt_len = self.config.packet_length

        for _ in range(num_pairs):
            r1 = self._run_single_trial(snr_db, n_bits, pkt_len)
            r2 = self._run_single_trial(snr_db, n_bits, pkt_len)
            # 取平均作為 antithetic 估計
            ber_list.append((r1["ber"] + r2["ber"]) / 2)

        agg = self.perf.aggregate_results(ber_list)
        agg["snr_db"] = snr_db
        agg["method"] = "antithetic"
        return agg
