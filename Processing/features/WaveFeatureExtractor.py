from Processing.base import *
from typing import Dict
@dataclass
class WaveFeatures:
    """Container for wave morphological features"""
    onset: int
    offset: int
    peak: int
    duration: float
    amplitude: float
    area: float
    slopes: Dict[str, float]
    energy: float


class WaveFeatureExtractor:
    """Extract morphological features from ECG waves"""

    def __init__(self, fs: float):
        self.fs = fs

    def extract_wave_features(self, signal: np.ndarray,
                              wave_type: str,
                              points: Dict[str, int]) -> WaveFeatures:
        """
        Extract features from wave segment

        Args:
            signal: Input wave segment
            wave_type: Type of wave ('P', 'QRS', or 'T')
            points: Dictionary containing wave characteristic points

        Returns:
            WaveFeatures object containing extracted features
        """

        if wave_type == 'QRS':
            return self._extract_qrs_features(signal, points)
        else:
            return self._extract_pt_features(signal, points, wave_type)

    def _extract_qrs_features(self, wave: np.ndarray,
                              points: Dict[str, int]) -> WaveFeatures:
        """Extract features specific to QRS complex"""

        # Extract basic time points
        onset = points['Q_onset'] if points['Q_onset'] is not None else 0
        offset = points['S_offset'] if points['S_offset'] is not None else len(wave) - 1

        # Calculate baseline as mean of onset region
        baseline_window = 10
        baseline = np.mean(wave[max(0, onset - baseline_window):onset + 1])

        # Get key points and intervals
        r_peak = points['R']
        q_point = points['Q']
        s_point = points['S']

        # Calculate durations
        duration = (offset - onset) / self.fs * 1000  # in ms

        # Calculate amplitudes relative to baseline
        r_amp = wave[r_peak] - baseline
        q_amp = wave[q_point] - baseline if q_point is not None else 0
        s_amp = wave[s_point] - baseline if s_point is not None else 0
        amplitude = max(abs(r_amp), abs(q_amp), abs(s_amp))

        # Calculate slopes
        slopes = {}
        if q_point is not None:
            slopes['Q-R'] = self._calculate_slope(wave, q_point, r_peak)
        if s_point is not None:
            slopes['R-S'] = self._calculate_slope(wave, r_peak, s_point)

        # Calculate area and energy
        area = np.trapz(np.abs(wave[onset:offset + 1] - baseline))
        energy = np.sum((wave[onset:offset + 1] - baseline) ** 2)

        return WaveFeatures(
            onset=onset,
            offset=offset,
            peak=r_peak,
            duration=duration,
            amplitude=amplitude,
            area=area,
            slopes=slopes,
            energy=energy
        )

    def _extract_pt_features(self, wave: np.ndarray,
                             points: Dict[str, int],
                             wave_type: str) -> WaveFeatures:
        """Extract features from P or T wave"""

        # Extract basic time points
        onset = points['onset'] if points['onset'] is not None else 0
        offset = points['offset'] if points['offset'] is not None else len(wave) - 1
        peak = points['peak'] if points['peak'] is not None else (onset + offset) // 2

        # Calculate baseline
        baseline_window = 10
        baseline = np.mean(wave[max(0, onset - baseline_window):onset + 1])

        # Calculate durations
        duration = (offset - onset) / self.fs * 1000  # in ms

        # Calculate amplitude
        amplitude = wave[peak] - baseline

        # Calculate slopes for ascending and descending parts
        slopes = {
            'ascending': self._calculate_slope(wave, onset, peak),
            'descending': self._calculate_slope(wave, peak, offset)
        }

        # Calculate maximum slope
        derivative = np.diff(wave[onset:offset + 1]) * self.fs
        slopes['max'] = np.max(np.abs(derivative))

        # Calculate area and energy
        area = np.trapz(np.abs(wave[onset:offset + 1] - baseline))
        energy = np.sum((wave[onset:offset + 1] - baseline) ** 2)

        return WaveFeatures(
            onset=onset,
            offset=offset,
            peak=peak,
            duration=duration,
            amplitude=amplitude,
            area=area,
            slopes=slopes,
            energy=energy
        )

    def _calculate_slope(self, signal: np.ndarray,
                         start_idx: int, end_idx: int) -> float:
        """Calculate slope between two points in signal"""
        if start_idx == end_idx:
            return 0.0

        time_diff = (end_idx - start_idx) / self.fs
        amplitude_diff = signal[end_idx] - signal[start_idx]

        return amplitude_diff / time_diff


class WaveAnalyzer:
    """Analyze wave morphology and extract clinical parameters"""

    def __init__(self, fs: float):
        self.fs = fs
        self.feature_extractor = WaveFeatureExtractor(fs)

    def analyze_beat(self, beat: np.ndarray,
                     points: Dict[str, Dict[str, int]]) -> Dict[str, Dict]:
        """
        Analyze complete heartbeat and extract features for all waves

        Args:
            beat: Input ECG beat
            points: Dictionary containing points for all waves

        Returns:
            Dictionary containing features and clinical parameters
        """

        # Extract features for each wave
        features = {}
        for wave_type in ['P', 'QRS', 'T']:
            if wave_type in points:
                features[wave_type] = self.feature_extractor.extract_wave_features(
                    beat, wave_type, points[wave_type]
                )

        # Calculate intervals
        intervals = self._calculate_intervals(points)

        # Combine features and intervals
        return {
            'features': features,
            'intervals': intervals
        }

    def _calculate_intervals(self, points: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """Calculate clinical intervals between waves"""
        intervals = {}

        # PR interval
        if 'P' in points and 'QRS' in points:
            if points['P']['onset'] is not None and points['QRS']['Q_onset'] is not None:
                intervals['PR'] = (points['QRS']['Q_onset'] - points['P']['onset']) / self.fs * 1000

        # QT interval
        if 'QRS' in points and 'T' in points:
            if points['QRS']['Q_onset'] is not None and points['T']['offset'] is not None:
                intervals['QT'] = (points['T']['offset'] - points['QRS']['Q_onset']) / self.fs * 1000

        # Calculate corrected QT (QTc) using Bazett's formula
        if 'QT' in intervals and 'RR' in points:
            rr_interval = points['RR'] / self.fs
            intervals['QTc'] = intervals['QT'] / np.sqrt(rr_interval)

        return intervals