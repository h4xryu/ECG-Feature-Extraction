from Processing.base import *

def plot_results(original: np.ndarray, reconstructed: np.ndarray,
                 delineation: dict, fs: float):
    """Plot original and reconstructed signals with delineation points"""
    time = np.arange(len(original)) / fs

    plt.figure(figsize=(12, 8))
    plt.plot(time, original, 'b', label='Original')
    plt.plot(time, reconstructed, 'r', label='Reconstructed')

    # Plot delineation points
    for wave, points in delineation.items():
        for point_type, idx in points.items():
            plt.axvline(x=idx / fs, color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()