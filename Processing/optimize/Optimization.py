from Processing.config.ECGConfig import *

@dataclass
class SignalQualityMetrics:
    """Container for signal quality metrics"""
    snr: float  # Signal-to-Noise Ratio
    prd: float  # Percentage Root mean square Difference
    rmse: float  # Root Mean Square Error
    correlation: float  # Correlation coefficient with original signal


class SignalQualityAnalyzer:
    """Analyze ECG signal quality and reconstruction accuracy"""

    def __init__(self, fs: float):
        self.fs = fs

    def compute_metrics(self, original: np.ndarray,
                        reconstructed: np.ndarray) -> SignalQualityMetrics:
        """Compute quality metrics between original and reconstructed signals"""
        # Remove mean from signals
        original = original - np.mean(original)
        reconstructed = reconstructed - np.mean(reconstructed)

        # Compute SNR
        noise = original - reconstructed
        signal_power = np.sum(original ** 2)
        noise_power = np.sum(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        # Compute PRD
        prd = np.sqrt(noise_power / signal_power) * 100

        # Compute RMSE
        rmse = np.sqrt(np.mean(noise ** 2))

        # Compute correlation
        correlation = stats.pearsonr(original, reconstructed)[0]

        return SignalQualityMetrics(snr, prd, rmse, correlation)

    def assess_beat_quality(self, beat: np.ndarray) -> bool:
        """Assess if beat quality is good enough for analysis"""
        # Check for flat lines
        if np.std(beat) < 1e-6:
            return False

        # Check for excessive noise using moving standard deviation
        window_size = int(0.05 * self.fs)  # 50ms window
        moving_std = np.array([np.std(beat[i:i + window_size])
                               for i in range(len(beat) - window_size)])
        if np.max(moving_std) / np.min(moving_std) > 10:  # High variation in noise level
            return False

        # Check for missing data
        if np.any(np.isnan(beat)):
            return False

        return True


class OptimizationManager:
    """Manage optimization of ECG signal decomposition"""

    def __init__(self, config: ECGConfig):
        self.config = config
        self.current_metrics = None

    def optimize_parameters(self, beat: np.ndarray,
                            initial_params: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Optimize decomposition parameters for a single beat"""
        from scipy.optimize import minimize

        # Define optimization bounds
        bounds = self._get_parameter_bounds(beat)

        # Optimize using L-BFGS-B
        result = minimize(
            fun=self._objective_function,
            x0=initial_params,
            args=(beat,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )

        # Store metrics
        self.current_metrics = self._calculate_metrics(beat, result.x)

        return result.x, {
            'success': result.success,
            'iterations': result.nit,
            'function_calls': result.nfev,
            'metrics': self.current_metrics
        }

    def _objective_function(self, params: np.ndarray,
                            beat: np.ndarray) -> float:
        """Objective function for optimization"""
        reconstructed = self._reconstruct_beat(params, beat)
        error = np.sum((beat - reconstructed) ** 2)

        # Add regularization terms
        smoothness_penalty = self._smoothness_penalty(reconstructed)
        sparsity_penalty = self._sparsity_penalty(params)

        return error + 0.1 * smoothness_penalty + 0.01 * sparsity_penalty

    def _smoothness_penalty(self, signal: np.ndarray) -> float:
        """Calculate smoothness penalty using second derivative"""
        second_derivative = np.gradient(np.gradient(signal))
        return np.sum(second_derivative ** 2)

    def _sparsity_penalty(self, params: np.ndarray) -> float:
        """Calculate sparsity penalty using L1 norm"""
        return np.sum(np.abs(params))

    def _get_parameter_bounds(self, beat: np.ndarray) -> List[Tuple[float, float]]:
        """Get bounds for optimization parameters"""
        beat_range = np.max(beat) - np.min(beat)
        bounds = []

        # Bounds for QRS parameters
        bounds.extend([(-beat_range, beat_range)] * self.config.hermite_nums[0])  # QRS amplitudes
        bounds.extend([(0.1, 10)] * 2)  # QRS width and position

        # Bounds for T wave parameters
        bounds.extend([(-beat_range / 2, beat_range / 2)] * self.config.hermite_nums[1])  # T amplitudes
        bounds.extend([(0.1, 10)] * 2)  # T width and position

        if self.config.p_wave:
            # Bounds for P wave parameters
            bounds.extend([(-beat_range / 3, beat_range / 3)] * self.config.hermite_nums[2])  # P amplitudes
            bounds.extend([(0.1, 10)] * 2)  # P width and position

        return bounds

    def _calculate_metrics(self, beat: np.ndarray,
                           params: np.ndarray) -> dict:
        """Calculate quality metrics for current optimization"""
        reconstructed = self._reconstruct_beat(params, beat)
        analyzer = SignalQualityAnalyzer(self.config.fs)
        metrics = analyzer.compute_metrics(beat, reconstructed)

        return {
            'snr': metrics.snr,
            'prd': metrics.prd,
            'rmse': metrics.rmse,
            'correlation': metrics.correlation,
            'params_l1': np.sum(np.abs(params)),
            'params_l2': np.sum(params ** 2)
        }

    def _reconstruct_beat(self, params: np.ndarray,
                          beat: np.ndarray) -> np.ndarray:
        """Reconstruct beat using current parameters"""
        # Implementation depends on the specific decomposition method used
        # This should be implemented based on your Hermite function implementation
        pass


class AdaptiveThresholder:
    """Adaptive thresholding for wave delineation"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.threshold_history = []

    def compute_threshold(self, signal: np.ndarray) -> float:
        """Compute adaptive threshold based on signal statistics"""
        # Compute signal statistics in window
        signal_std = np.std(signal)
        signal_range = np.ptp(signal)

        # Base threshold on noise level
        base_threshold = 3 * signal_std

        # Adjust threshold based on signal range
        adjusted_threshold = min(base_threshold, 0.2 * signal_range)

        # Update threshold history
        self.threshold_history.append(adjusted_threshold)
        if len(self.threshold_history) > self.window_size:
            self.threshold_history.pop(0)

        # Use median of recent thresholds for stability
        return np.median(self.threshold_history)

    def reset(self):
        """Reset threshold history"""
        self.threshold_history = []