from Processing.base import *
from scipy.signal import find_peaks, savgol_filter, medfilt, butter, filtfilt
from typing import Dict, Optional, Tuple, List

class WaveDelineator:
    def __init__(self, fs: float):
        self.fs = fs
        # Parameters for optimization
        self.wave_params = {
            'QRS': {'hermite_funcs': 7, 'use_sigmoid': True},
            'T': {'hermite_funcs': 6, 'use_sigmoid': True},
            'P': {'hermite_funcs': 4, 'use_sigmoid': False},
        }
        # Constraints based on paper
        self.constraints = {
            'QRS': {'width': (0.06, 0.12)},  # 60-120ms
            'T': {'width': (0.1, 0.25)},  # 100-250ms
            'P': {'width': (0.08, 0.125)}  # 80-125ms
        }

    def delineate_beat(self, beat: np.ndarray) -> Dict:
        """Main method for beat delineation using variable projection"""
        # 1. Remove baseline using spline interpolation
        baseline = self._estimate_baseline(beat)
        beat_corrected = beat - baseline

        # 2. Segment waves using variable projection
        waves = {}

        # QRS first
        qrs_result = self._segment_wave(beat_corrected, 'QRS')
        waves['QRS'] = qrs_result

        # T wave after QRS
        if qrs_result['offset'] is not None:
            t_segment = beat_corrected[qrs_result['offset']:]
            t_result = self._segment_wave(t_segment, 'T')
            # Adjust indices
            t_result = {k: (v + qrs_result['offset'] if v is not None else None)
                        for k, v in t_result.items()}
            waves['T'] = t_result
        else:
            waves['T'] = {'onset': None, 'peak': None, 'offset': None}

        # P wave before QRS
        if qrs_result['onset'] is not None:
            p_segment = beat_corrected[:qrs_result['onset']]
            p_result = self._segment_wave(p_segment, 'P')
            waves['P'] = p_result
        else:
            waves['P'] = {'onset': None, 'peak': None, 'offset': None}

        return waves

    def _estimate_baseline(self, beat: np.ndarray) -> np.ndarray:
        """Estimate baseline using cubic spline interpolation"""
        from scipy.interpolate import PchipInterpolator

        # Define knots at PQ and TP segments
        n_samples = len(beat)
        knots = [0, n_samples // 4, 3 * n_samples // 4, n_samples - 1]

        # Get values at knots
        knot_values = [beat[k] for k in knots]

        # Interpolate using PCHIP (shape-preserving)
        interpolator = PchipInterpolator(knots, knot_values)
        baseline = interpolator(np.arange(n_samples))

        return baseline

    def _segment_wave(self, signal: np.ndarray, wave_type: str) -> Dict:
        """Segment specific wave using variable projection"""
        # 1. Create dictionary
        Phi = self._create_dictionary(signal, wave_type)

        # 2. Initial parameters
        params_init = self._get_initial_params(signal, wave_type)

        # 3. Optimize using variable projection
        result = self._optimize_wave_params(signal, Phi, params_init, wave_type)

        # 4. Get points from optimized parameters
        points = self._get_wave_points(result, signal, wave_type)

        return points

    def _create_dictionary(self, signal: np.ndarray, wave_type: str) -> np.ndarray:
        """Create dictionary for wave approximation"""
        n_samples = len(signal)
        t = np.linspace(-4, 4, n_samples)
        dictionary = []

        # Add Hermite functions
        n_funcs = self.wave_params[wave_type]['hermite_funcs']
        for j in range(n_funcs):
            scale = 1.11 ** j  # Scaling factor from paper
            hf = self._hermite_function(j, scale * t)
            dictionary.append(hf)

        # Add sigmoid if needed
        if self.wave_params[wave_type]['use_sigmoid']:
            sigmoid = 1 / (1 + np.exp(-2 * t))
            dictionary.append(sigmoid)

        return np.array(dictionary).T

    def _hermite_function(self, n: int, t: np.ndarray) -> np.ndarray:
        """Generate nth order Hermite function"""
        from scipy.special import hermite
        H = hermite(n)
        phi = H(t) * np.exp(-t ** 2 / 2)
        norm = np.sqrt(2 ** n * np.math.factorial(n) * np.sqrt(np.pi))
        return phi / norm

    def _optimize_wave_params(self, signal: np.ndarray, Phi: np.ndarray,
                              params_init: np.ndarray, wave_type: str) -> np.ndarray:
        """Optimize parameters using variable projection"""
        from scipy.optimize import minimize

        def objective(params):
            transformed_Phi = self._transform_dictionary(Phi, params)
            Phi_plus = np.linalg.pinv(transformed_Phi)
            residual = signal - transformed_Phi @ (Phi_plus @ signal)
            return np.sum(residual ** 2)

        # Add constraints from paper
        bounds = self._get_bounds(wave_type, len(signal))

        result = minimize(objective, params_init, bounds=bounds)
        return result.x

    def _get_wave_points(self, params: np.ndarray, signal: np.ndarray,
                         wave_type: str) -> Dict:
        """Get wave points from optimized parameters"""
        # Using 3-sigma rule from paper
        lambda_, tau = params[:2]
        onset = int(max(0, tau - 3 / lambda_))
        offset = int(min(len(signal) - 1, tau + 3 / lambda_))

        # Find peak
        peak_segment = signal[onset:offset]
        peak = onset + np.argmax(np.abs(peak_segment))

        return {
            'onset': onset,
            'peak': peak,
            'offset': offset
        }