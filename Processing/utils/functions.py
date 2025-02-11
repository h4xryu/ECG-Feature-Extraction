
from Processing.base import *

def generate_hermite_functions(n: int, sigma: float,
                               t: np.ndarray) -> np.ndarray:
    """Generate Hermite functions for signal approximation"""
    H = hermite(n)
    x = t / sigma
    phi = H(x) * np.exp(-x ** 2 / 2)
    return phi / np.sqrt(2 ** n * np.math.factorial(n) * np.sqrt(np.pi))


def find_wave_boundaries(wave: np.ndarray,
                         threshold: float = 0.1) -> Tuple[int, int, int]:
    """Find wave onset, peak, and offset"""
    # Calculate first derivative
    derivative = np.gradient(wave)

    # Find peaks in derivative
    pos_peaks, _ = find_peaks(derivative)
    neg_peaks, _ = find_peaks(-derivative)

    if len(pos_peaks) == 0 or len(neg_peaks) == 0:
        return 0, 0, 0

    # Find main peak
    peak_idx = np.argmax(np.abs(wave))

    # Find onset using derivative threshold
    for i in range(peak_idx, 0, -1):
        if abs(derivative[i]) < threshold:
            onset = i
            break
    else:
        onset = 0

    # Find offset using derivative threshold
    for i in range(peak_idx, len(wave)):
        if abs(derivative[i]) < threshold:
            offset = i
            break
    else:
        offset = len(wave) - 1

    return onset, peak_idx, offset


def baseline_correction(signal: np.ndarray,
                        window_size: int = 200) -> np.ndarray:
    """Remove baseline wander using moving average"""
    padding = np.ones(window_size)
    padding *= signal[0]

    signal_padded = np.concatenate([padding, signal, padding])

    # Calculate moving average
    window = np.ones(window_size) / window_size
    baseline = np.convolve(signal_padded, window, mode='same')

    # Remove padding from baseline
    baseline = baseline[window_size:-window_size]

    return signal - baseline


def sigmoid(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Generate sigmoid function"""
    return 1 / (1 + np.exp(-alpha * (x - beta)))


def detect_fiducial_points(wave: np.ndarray,
                           fs: float) -> dict:
    """Detect characteristic points in ECG wave"""
    # Calculate first and second derivatives
    d1 = np.gradient(wave)
    d2 = np.gradient(d1)

    # Find positive and negative peaks
    pos_peaks, _ = find_peaks(wave)
    neg_peaks, _ = find_peaks(-wave)

    # Find R peak (maximum absolute amplitude)
    r_peak = np.argmax(np.abs(wave))

    # Find Q and S points
    q_point = None
    s_point = None

    for peak in reversed(neg_peaks):
        if peak < r_peak:
            q_point = peak
            break

    for peak in neg_peaks:
        if peak > r_peak:
            s_point = peak
            break

    # Create result dictionary
    points = {
        'Q': q_point,
        'R': r_peak,
        'S': s_point,
        'Q_onset': None,
        'S_offset': None
    }

    # Find QRS onset and offset using derivative threshold
    if q_point is not None:
        for i in range(q_point, 0, -1):
            if abs(d1[i]) < 0.1 * max(abs(d1)):
                points['Q_onset'] = i
                break

    if s_point is not None:
        for i in range(s_point, len(wave)):
            if abs(d1[i]) < 0.1 * max(abs(d1)):
                points['S_offset'] = i
                break

    return points


def calculate_wave_features(wave: np.ndarray,
                            points: dict,
                            fs: float) -> dict:
    """Calculate various features of the wave"""
    features = {}

    # Time intervals
    if points['Q'] is not None and points['S'] is not None:
        features['QRS_duration'] = (points['S'] - points['Q']) / fs * 1000  # in ms

    if points['Q_onset'] is not None and points['S_offset'] is not None:
        features['QRS_total_duration'] = (points['S_offset'] - points['Q_onset']) / fs * 1000

    # Amplitudes
    baseline = np.mean(wave[:points['Q_onset']] if points['Q_onset'] is not None else wave[:10])

    if points['Q'] is not None:
        features['Q_amplitude'] = wave[points['Q']] - baseline

    features['R_amplitude'] = wave[points['R']] - baseline

    if points['S'] is not None:
        features['S_amplitude'] = wave[points['S']] - baseline

    # Calculate additional features
    features['QRS_area'] = np.trapz(wave[points['Q_onset']:points['S_offset']]) if (
                points['Q_onset'] is not None and points['S_offset'] is not None) else None

    # Calculate slopes
    if points['Q'] is not None and points['R'] is not None:
        features['QR_slope'] = (wave[points['R']] - wave[points['Q']]) / ((points['R'] - points['Q']) / fs)

    if points['R'] is not None and points['S'] is not None:
        features['RS_slope'] = (wave[points['S']] - wave[points['R']]) / ((points['S'] - points['R']) / fs)

    # R wave symmetry
    if points['Q'] is not None and points['S'] is not None:
        r_peak_pos = (points['R'] - points['Q']) / (points['S'] - points['Q'])
        features['R_symmetry'] = abs(0.5 - r_peak_pos)  # 0 means perfect symmetry

    return features