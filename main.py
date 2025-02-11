from Processing import *
from scipy.io import loadmat
from pathlib import Path
import json


def main():
    # Load configuration
    config = {
        'fs': 250,  # Sampling frequency
        'p_wave': True,
        'u_wave': False,
        'hermite_nums': [7, 6, 4],  # Number of Hermite functions for QRS, T, P waves
        'rules': [3, 3, 3]  # Optimization constraints
    }

    # Load ECG data
    data = loadmat('ecg_data.mat')
    signal = data['xx'].squeeze()
    qrs_locs = data['QRS_locs'].squeeze()
    fs = float(data['fs'][0][0])

    # Initialize processors
    ecg_config = ECGConfig(fs=fs)
    processor = ECGProcessor(ecg_config)
    delineator = WaveDelineator(fs)
    hermite_fitter = HermiteFitter(fs)

    # Process signal
    print("Preparing data...")
    mean_beat, test_beats = processor.prepare_data(signal, qrs_locs)

    print("Extracting features...")
    coeffs, opt_pars = processor.extract_features(test_beats)

    # Process each beat
    results = []
    for i, beat in enumerate(test_beats):
        print(f"Processing beat {i + 1}/{len(test_beats)}")

        # Delineate waves
        delineation = delineator.delineate_beat(beat)

        # Fit Hermite functions
        qrs_coeffs, qrs_recon = hermite_fitter.fit_wave(
            beat[delineation['QRS']['onset']:delineation['QRS']['offset']],
            config['hermite_nums'][0]
        )

        t_coeffs, t_recon = hermite_fitter.fit_wave(
            beat[delineation['T']['onset']:delineation['T']['offset']],
            config['hermite_nums'][1]
        )

        if delineation['P']['peak'] is not None:
            p_coeffs, p_recon = hermite_fitter.fit_wave(
                beat[delineation['P']['onset']:delineation['P']['offset']],
                config['hermite_nums'][2]
            )
        else:
            p_coeffs = np.zeros(config['hermite_nums'][2])
            p_recon = np.array([])

        # Store results
        results.append({
            'delineation': delineation,
            'coefficients': {
                'qrs': qrs_coeffs.tolist(),
                't': t_coeffs.tolist(),
                'p': p_coeffs.tolist()
            },
            'reconstruction': {
                'qrs': qrs_recon.tolist(),
                't': t_recon.tolist(),
                'p': p_recon.tolist() if len(p_recon) > 0 else []
            }
        })

        # Plot every 10th beat
        if i % 10 == 0:
            plot_beat_analysis(beat, delineation, qrs_recon, t_recon, p_recon, fs, i)

    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/analysis_results.json', 'w') as f:
        json.dump(results, f)

    print("Processing complete. Results saved to 'results/analysis_results.json'")


def plot_beat_analysis(beat, delineation, qrs_recon, t_recon, p_recon, fs, beat_num):
    """Plot original beat with delineation and reconstructed waves"""
    t = np.arange(len(beat)) / fs

    plt.figure(figsize=(15, 10))

    # Plot original signal
    plt.plot(t, beat, 'k-', label='Original', alpha=0.6)

    # Plot reconstructed waves
    if delineation['QRS']['onset'] is not None:
        t_qrs = np.arange(len(qrs_recon)) / fs + delineation['QRS']['onset'] / fs
        plt.plot(t_qrs, qrs_recon, 'r-', label='QRS reconstruction')

    if delineation['T']['onset'] is not None:
        t_t = np.arange(len(t_recon)) / fs + delineation['T']['onset'] / fs
        plt.plot(t_t, t_recon, 'g-', label='T reconstruction')

    if len(p_recon) > 0:
        t_p = np.arange(len(p_recon)) / fs + delineation['P']['onset'] / fs
        plt.plot(t_p, p_recon, 'b-', label='P reconstruction')

    # Plot delineation points
    for wave, points in delineation.items():
        for point_type, idx in points.items():
            if idx is not None:
                plt.axvline(x=idx / fs, color='gray', linestyle='--', alpha=0.3)
                plt.text(idx / fs, plt.ylim()[0], f'{wave}_{point_type}',
                         rotation=90, verticalalignment='bottom')

    plt.title(f'Beat Analysis - Beat #{beat_num}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(f'results/beat_{beat_num}_analysis.png')
    plt.close()


if __name__ == "__main__":
    main()