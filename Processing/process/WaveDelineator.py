from Processing.base import *


class WaveDelineator:
    """Wave delineation processor for ECG signals"""

    def __init__(self, fs: float):
        self.fs = fs
        self.thresholds = {
            'p_wave': 0.25,
            'qrs': 0.1,
            't_wave': 0.4
        }

    def delineate_beat(self, beat: np.ndarray) -> dict:
        """Perform complete delineation of a single beat"""
        # Normalize beat
        beat_norm = (beat - np.mean(beat)) / np.std(beat)

        # Find QRS complex
        qrs_points = self._delineate_qrs(beat_norm)

        # Find P wave
        p_points = self._delineate_p_wave(beat_norm[:qrs_points['onset']])
        if p_points['peak'] is not None:
            p_points = {k: v + qrs_points['onset'] if v is not None else None
                        for k, v in p_points.items()}

        # Find T wave
        t_points = self._delineate_t_wave(beat_norm[qrs_points['offset']:])
        if t_points['peak'] is not None:
            t_points = {k: v + qrs_points['offset'] if v is not None else None
                        for k, v in t_points.items()}

        return {
            'P': p_points,
            'QRS': qrs_points,
            'T': t_points
        }

    def _delineate_qrs(self, signal: np.ndarray) -> dict:
        """Delineate QRS complex"""
        # Calculate derivatives
        d1 = np.gradient(signal)
        d2 = np.gradient(d1)

        # Find R peak
        r_peak = np.argmax(np.abs(signal))

        # Find Q and S points
        q_valley = None
        s_valley = None

        # Search for Q point
        possible_q = np.where((d1[:r_peak] > 0) & (d2[:r_peak] > 0))[0]
        if len(possible_q) > 0:
            q_valley = possible_q[-1]

        # Search for S point
        possible_s = np.where((d1[r_peak:] < 0) & (d2[r_peak:] < 0))[0]
        if len(possible_s) > 0:
            s_valley = possible_s[0] + r_peak

        # Find onset using slope threshold
        onset = self._find_wave_onset(signal[:q_valley] if q_valley is not None else signal[:r_peak],
                                      self.thresholds['qrs'])

        # Find offset using slope threshold
        offset = self._find_wave_offset(signal[s_valley:] if s_valley is not None else signal[r_peak:],
                                        self.thresholds['qrs'])
        offset = offset + (s_valley if s_valley is not None else r_peak)

        return {
            'onset': onset,
            'Q': q_valley,
            'R': r_peak,
            'S': s_valley,
            'offset': offset
        }

    def _delineate_p_wave(self, signal: np.ndarray) -> dict:
        """Delineate P wave"""
        if len(signal) < 3:
            return {'onset': None, 'peak': None, 'offset': None}

        # Find peak
        peaks, _ = find_peaks(signal, distance=int(0.05 * self.fs))
        if len(peaks) == 0:
            return {'onset': None, 'peak': None, 'offset': None}

        peak = peaks[-1]  # Take the last peak as P wave

        # Find onset and offset
        onset = self._find_wave_onset(signal[:peak], self.thresholds['p_wave'])
        offset = self._find_wave_offset(signal[peak:], self.thresholds['p_wave'])
        offset = offset + peak if offset is not None else None

        return {
            'onset': onset,
            'peak': peak,
            'offset': offset
        }

    def _delineate_t_wave(self, signal: np.ndarray) -> dict:
        """Delineate T wave"""
        if len(signal) < 3:
            return {'onset': None, 'peak': None, 'offset': None}

        # Find peak
        peaks, _ = find_peaks(signal, distance=int(0.05 * self.fs))
        if len(peaks) == 0:
            return {'onset': None, 'peak': None, 'offset': None}

        peak = peaks[0]  # Take the first peak as T wave

        # Find onset and offset
        onset = self._find_wave_onset(signal[:peak], self.thresholds['t_wave'])
        offset = self._find_wave_offset(signal[peak:], self.thresholds['t_wave'])
        offset = offset + peak if offset is not None else None

        return {
            'onset': onset,
            'peak': peak,
            'offset': offset
        }

    @staticmethod
    def _find_wave_onset(signal: np.ndarray, threshold: float) -> Optional[int]:
        """Find wave onset using derivative threshold"""
        if len(signal) < 2:
            return None

        derivative = np.gradient(signal)
        max_slope = np.max(np.abs(derivative))

        for i in range(len(signal) - 1, 0, -1):
            if abs(derivative[i]) < threshold * max_slope:
                return i
        return 0

    @staticmethod
    def _find_wave_offset(signal: np.ndarray, threshold: float) -> Optional[int]:
        """Find wave offset using derivative threshold"""
        if len(signal) < 2:
            return None

        derivative = np.gradient(signal)
        max_slope = np.max(np.abs(derivative))

        for i in range(len(signal)):
            if abs(derivative[i]) < threshold * max_slope:
                return i
        return len(signal) - 1

