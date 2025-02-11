from Processing.base import *

class HermiteFitter:
    """Fit Hermite functions to ECG waves"""

    def __init__(self, fs: float):
        self.fs = fs

    def fit_wave(self, wave: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fit Hermite functions to a wave segment"""
        # Normalize time vector
        t = np.linspace(-1, 1, len(wave))

        # Initialize matrices
        H = np.zeros((len(t), n_components))

        # Generate Hermite functions
        for n in range(n_components):
            H[:, n] = generate_hermite_functions(n, 1.0, t)

        # Solve for coefficients using least squares
        coeffs = np.linalg.lstsq(H, wave, rcond=None)[0]

        # Reconstruct signal
        reconstruction = H @ coeffs

        return coeffs, reconstruction