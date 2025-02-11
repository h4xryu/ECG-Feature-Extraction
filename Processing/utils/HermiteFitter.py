from Processing.base import *


class HermiteFitter:
    """Fit Hermite functions to ECG waves"""

    def __init__(self, fs: float):
        self.fs = fs

    def fit_wave(self, wave: np.ndarray, n_components: int) -> tuple:
        """Fit Hermite functions to a wave segment"""
        # Normalize time vector
        t = np.linspace(-1, 1, len(wave))

        # Initialize matrices
        H = np.zeros((len(t), n_components))

        # Generate Hermite functions
        for n in range(n_components):
            H[:, n] = self._generate_hermite_functions(n, 1.0, t)

        # Solve for coefficients using least squares
        coeffs = np.linalg.lstsq(H, wave, rcond=None)[0]

        # Reconstruct signal
        reconstruction = H @ coeffs

        return coeffs, reconstruction

    def _generate_hermite_functions(self, n: int, sigma: float, t: np.ndarray) -> np.ndarray:
        """Generate nth order Hermite function

        Args:
            n: Order of Hermite polynomial
            sigma: Scale parameter
            t: Time points

        Returns:
            Hermite function values
        """
        # Generate Hermite polynomial
        Hn = hermite(n)
        x = t / sigma

        # Multiply by Gaussian envelope and normalize
        phi = Hn(x) * np.exp(-x ** 2 / 2)
        norm = np.sqrt(2 ** n * np.math.factorial(n) * np.sqrt(np.pi))

        return phi / norm