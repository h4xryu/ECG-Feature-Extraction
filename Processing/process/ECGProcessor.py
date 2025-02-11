from Processing.config.ECGConfig import *


class ECGProcessor:
    """Main class for ECG signal processing"""

    def __init__(self, config: ECGConfig):
        self.config = config
        self.mean_beat = None
        self.coeffs = None
        self.opt_pars = None

        # Ensure hermite_nums is properly initialized
        if self.config.hermite_nums is None:
            if self.config.p_wave and self.config.u_wave:
                self.config.hermite_nums = [7, 6, 4, 2]  # [QRS, T, P, U]
            elif self.config.p_wave:
                self.config.hermite_nums = [7, 6, 4]  # [QRS, T, P]
            elif self.config.u_wave:
                self.config.hermite_nums = [7, 6, 2]  # [QRS, T, U]
            else:
                self.config.hermite_nums = [7, 6]  # [QRS, T]

    def prepare_data(self, signal: np.ndarray, qrs_locs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Prepare training and test sets from ECG signal"""
        # Take first 100 beats for training
        train_idx = np.arange(100)

        # Calculate RR intervals
        rr_train = np.diff(qrs_locs[train_idx]).mean()

        if self.config.pre_r is None:
            self.config.pre_r = int(rr_train / 3)
        post_r = int(rr_train - self.config.pre_r)

        # Extract training beats
        train_beats = []
        for idx in train_idx:
            if qrs_locs[idx] >= self.config.pre_r:
                beat = signal[qrs_locs[idx] - self.config.pre_r:
                              qrs_locs[idx] + post_r]
                train_beats.append(beat)

        # Calculate mean beat
        self.mean_beat = np.mean(train_beats, axis=0)

        # Extract test beats
        test_beats = []
        test_idx = range(100, len(qrs_locs) - 1)
        for idx in test_idx:
            beat = signal[qrs_locs[idx] - self.config.pre_r:
                          qrs_locs[idx + 1] - self.config.pre_r]
            test_beats.append(beat)

        return self.mean_beat, test_beats

    def extract_features(self, beats: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features using variable projection method"""
        # Calculate total number of parameters
        num_params = 0
        # Parameters for QRS and T waves (always present)
        num_params += (self.config.hermite_nums[0] + 2)  # QRS coefficients + width & position
        num_params += (self.config.hermite_nums[1] + 2)  # T wave coefficients + width & position

        # Parameters for P wave if present
        if self.config.p_wave:
            num_params += (self.config.hermite_nums[2] + 2)

        # Parameters for U wave if present
        if self.config.u_wave:
            wave_idx = 3 if self.config.p_wave else 2
            num_params += (self.config.hermite_nums[wave_idx] + 2)

        # Initialize matrices for coefficients and parameters
        num_beats = len(beats)
        self.coeffs = np.zeros((num_params, num_beats))
        self.opt_pars = np.zeros((num_params, num_beats))

        # Global optimization for mean beat
        opt_pars_init = self._global_optimization(self.mean_beat)

        # Local optimization for each beat
        for i, beat in enumerate(beats):
            coeffs, pars = self._local_optimization(beat, opt_pars_init)
            self.coeffs[:, i] = coeffs
            self.opt_pars[:, i] = pars

        return self.coeffs, self.opt_pars

    def _global_optimization(self, beat: np.ndarray) -> np.ndarray:
        """Perform global optimization using differential evolution"""
        bounds = self._get_optimization_bounds(beat)

        result = differential_evolution(
            func=self._cost_function,
            bounds=bounds,
            args=(beat,),
            maxiter=20,
            popsize=20
        )

        return result.x

    def _local_optimization(self, beat: np.ndarray, init_pars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform local optimization using BFGS"""
        result = minimize(
            fun=self._cost_function,
            x0=init_pars,
            args=(beat,),
            method='L-BFGS-B',
            bounds=self._get_optimization_bounds(beat)
        )

        # Get coefficients from optimized parameters
        reconstructed = self._reconstruct_beat(result.x, beat)
        coeffs = result.x  # In this case, coefficients are the same as parameters

        return coeffs, result.x

    def _cost_function(self, params: np.ndarray, beat: np.ndarray) -> float:
        """Cost function for optimization"""
        # Implement the least squares cost function
        reconstructed = self._reconstruct_beat(params, beat)
        return np.sum((beat - reconstructed) ** 2)

    def _get_optimization_bounds(self, beat: np.ndarray) -> List[Tuple[float, float]]:
        """Get bounds for optimization parameters"""
        # Get signal range for amplitude bounds
        signal_range = np.ptp(beat)
        bounds = []

        # Add bounds for each wave component
        if self.config.p_wave:
            # P wave bounds (amplitude and timing)
            bounds.extend([(-signal_range / 3, signal_range / 3)] * self.config.hermite_nums[2])  # P wave amplitudes
            bounds.extend([(0.1, 0.3)])  # P wave width (in seconds)
            bounds.extend([(0, len(beat) / 3)])  # P wave position

        # QRS bounds
        bounds.extend([(-signal_range, signal_range)] * self.config.hermite_nums[0])  # QRS amplitudes
        bounds.extend([(0.06, 0.12)])  # QRS width (in seconds)
        bounds.extend([(len(beat) / 3, 2 * len(beat) / 3)])  # QRS position

        # T wave bounds
        bounds.extend([(-signal_range / 2, signal_range / 2)] * self.config.hermite_nums[1])  # T wave amplitudes
        bounds.extend([(0.1, 0.25)])  # T wave width (in seconds)
        bounds.extend([(2 * len(beat) / 3, len(beat))])  # T wave position

        if self.config.u_wave:
            # U wave bounds
            bounds.extend([(-signal_range / 4, signal_range / 4)] * self.config.hermite_nums[3])  # U wave amplitudes
            bounds.extend([(0.1, 0.2)])  # U wave width (in seconds)
            bounds.extend([(2 * len(beat) / 3, len(beat))])  # U wave position

        return bounds

    def _get_coefficients(self, beat: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Calculate coefficients for the current parameter set"""
        # Implement coefficient calculation
        pass

    def _reconstruct_beat(self, params: np.ndarray, beat: np.ndarray) -> np.ndarray:
        """Reconstruct beat using current parameters"""
        reconstructed = np.zeros_like(beat)
        t = np.arange(len(beat))

        # Track current parameter index
        param_idx = 0

        # Add QRS complex
        qrs_params = params[param_idx:param_idx + self.config.hermite_nums[0] + 2]
        reconstructed += self._generate_wave(t, qrs_params, self.config.hermite_nums[0])
        param_idx += self.config.hermite_nums[0] + 2

        # Add T wave
        t_params = params[param_idx:param_idx + self.config.hermite_nums[1] + 2]
        reconstructed += self._generate_wave(t, t_params, self.config.hermite_nums[1])
        param_idx += self.config.hermite_nums[1] + 2

        # Add P wave if configured
        if self.config.p_wave:
            p_params = params[param_idx:param_idx + self.config.hermite_nums[2] + 2]
            reconstructed += self._generate_wave(t, p_params, self.config.hermite_nums[2])
            param_idx += self.config.hermite_nums[2] + 2

        # Add U wave if configured
        if self.config.u_wave:
            u_params = params[param_idx:param_idx + self.config.hermite_nums[3] + 2]
            reconstructed += self._generate_wave(t, u_params, self.config.hermite_nums[3])

        return reconstructed

    def _generate_wave(self, t: np.ndarray, params: np.ndarray, n_hermite: int) -> np.ndarray:
        """Generate single wave using Hermite functions"""
        # Extract shape parameters
        width = params[-2]
        position = params[-1]
        amplitudes = params[:-2]

        # Initialize wave
        wave = np.zeros_like(t, dtype=float)

        # Generate time vector centered at wave position
        t_normalized = (t - position) / (width * self.config.fs)

        # Add contribution from each Hermite function
        for n in range(n_hermite):
            hermite_basis = self._hermite_function(n, t_normalized)
            wave += amplitudes[n] * hermite_basis

        return wave

    def _hermite_function(self, n: int, x: np.ndarray) -> np.ndarray:
        """Generate nth order Hermite function"""
        # Generate Hermite polynomial
        coeffs = np.zeros(n + 1)
        coeffs[-1] = 1
        hermite = np.polynomial.hermite.hermval(x, coeffs)

        # Multiply by Gaussian envelope
        return hermite * np.exp(-x ** 2 / 2) / np.sqrt(2 ** n * np.math.factorial(n) * np.sqrt(np.pi))

    def _cost_function(self, params: np.ndarray, beat: np.ndarray) -> float:
        """Cost function for optimization"""
        # Reconstruct beat using current parameters
        reconstructed = self._reconstruct_beat(params, beat)

        # Calculate reconstruction error
        error = np.sum((beat - reconstructed) ** 2)

        # Add regularization for smoothness
        derivative = np.diff(reconstructed)
        smoothness_penalty = np.sum(derivative ** 2)

        # Add regularization for sparsity of parameters
        sparsity_penalty = np.sum(np.abs(params))

        return error + 0.1 * smoothness_penalty + 0.01 * sparsity_penalty