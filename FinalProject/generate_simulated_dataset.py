import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# set random seed
np.random.seed(42)


#********************#
#   initialization   #
#********************#

d = 1   # number of sources
m = 4   # number of array elements
snr = 10   # signal to noise ratio

# frequency range of interest
fs = np.linspace(0, 1000, 1000, endpoint=False)

# sampling frequency
fSamp = 2000

# distance between array elements = min wavelength / 2
dist = (3e8 / np.max(fs) + 1) / 2
# dist = 0.03

doa = np.pi * (np.random.rand(d) - 1/2)   # random source directions in [-pi/2, pi/2]
p = np.sqrt(1) * (np.random.randn(d) + np.random.randn(d) * 1j)    # random source powers

array = np.linspace(0, m, m, endpoint=False)   # array element positions
angles = np.array((np.linspace(- np.pi/2, np.pi/2, 360, endpoint=False),))   # angle continuum
# angles = np.array((np.linspace(0, 2 * np.pi, 720, endpoint=False),))   # angle continuum

snapshots = 200  # number of time samples
t = np.arange(0, 1, 1/fSamp)   # discrete time events


def ULA_steering_vector(array, theta, f, spacing):
    """
    Compute the steering vector for a uniform linear array.

    Parameters:
    - array: Positions of the array elements.
    - theta: Angle of arrival (radians).
    - f: Frequency of the signals.
    - spacing: Distance between array elements.

    Returns:
    - Steering vector.
    """
    return np.exp(-1j * 2 * np.pi * f * spacing * array * np.sin(theta) / 3e8)

def construct_ofdm_signal(thetas, fcs, num_subcarriers=1000, snr=20):
    """
    Construct an OFDM signal.

    Parameters:
    - thetas: Direction of arrival angles for sources.
    - fcs: Carrier frequencies of the sources.
    - num_subcarriers: Number of subcarriers per signal.
    - snr: Signal-to-noise ratio in dB.

    Returns:
    - OFDM signal.
    """
    d = len(thetas)
    signal = np.zeros((d, len(t))) + 1j * np.zeros((d, len(t)))

    for i in range(d):
        for j in range(num_subcarriers):
            amp = np.sqrt(2)/2 * (np.random.randn() + 1j * np.random.randn())
            signal[i] += amp * np.exp(1j * 2 * np.pi * j * len(fs) * t / num_subcarriers)

        signal[i] *= (10 ** (snr / 10)) * (1/num_subcarriers)

    noise = np.sqrt(2)/2 * (np.random.randn(d, len(t)) + 1j * np.random.randn(d, len(t)))
    signal = np.fft.fft(signal)
    noise = np.fft.fft(noise)

    X = []
    for i in range(int(fSamp)):
        if i > int(fSamp) // 2:
            f = - int(fSamp) + i
        else:
            f = i

        A = np.array([ULA_steering_vector(array, thetas[j], f, dist) for j in range(d)])
        X.append(np.dot(A.T, signal[:, i]) + noise[:, i])

    return np.fft.ifft(np.array(X).T, axis=1)[:, :snapshots]

def create_broadband_dataset(name, size, num_sources=[1], coherent=False, save=True):
    """
    Create a dataset for DoA estimation.

    Parameters:
    - name: Name of the file for the dataset.
    - size: Size of the dataset.
    - num_sources: Number of sources as a list.
    - coherent: If True, signals are coherent.
    - save: If True, save the dataset to a file.
    """
    X = np.zeros((size, m, snapshots)) + 1j * np.zeros((size, m, snapshots))
    Thetas = np.zeros((size, d))

    for i in tqdm(range(size)):
        num = num_sources[i % len(num_sources)]
        thetas = np.pi * (np.random.rand(num) - 1/2)
        fcs = np.random.choice(fs, d)

        if coherent:
            print("ERROR - not implemented yet!")
        else:
            X[i] = construct_ofdm_signal(thetas, fcs)

        Thetas[i] = thetas

    if save:
        import os
        if not (os.path.isfile('data/' + name + '.h5')):
            with open('data/' + name + '.h5', 'w'):
                pass
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Thetas)
        hf.close()

    return X, Thetas

def visualize_dataset_2d(dataset):
    num_samples, num_antennas, num_snapshots = dataset.shape
    num_sources = len(dataset[0])  # Assuming each sample can have a different number of sources

    # Set up the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 2D surface for the array response matrix (IQ components)
    ax1 = axes[0, 0]
    ax1.set_title(f'Array Response Matrix (IQ Components, Example 1 of {num_samples})')
    array_response_transposed = dataset[0].T
    c1 = ax1.imshow(np.real(array_response_transposed), cmap='viridis', aspect='auto', interpolation='none')
    ax1.set_xlabel('Antennas (X-axis)')
    ax1.set_ylabel('Antennas (Y-axis)')
    fig.colorbar(c1, ax=ax1, label='Amplitude')

    # Plot phase for the array response matrix
    ax2 = axes[0, 1]
    ax2.set_title('Array Response Matrix (Phase)')
    c2 = ax2.imshow(np.angle(array_response_transposed), cmap='hsv', aspect='auto', interpolation='none')
    ax2.set_xlabel('Antennas (X-axis)')
    ax2.set_ylabel('Antennas (Y-axis)')
    fig.colorbar(c2, ax=ax2, label='Phase (radians)')

    # Visualize observed signal for the first example (IQ components)
    ax3 = axes[1, 0]
    ax3.set_title(f'Observed Signal (IQ Components, Example 1 of {num_samples})')
    ax3.plot(np.real(dataset[0][:, 0]), label='Real')
    ax3.plot(np.imag(dataset[0][:, 0]), label='Imaginary')
    ax3.set_xlabel('Antennas')
    ax3.set_ylabel('Amplitude')
    ax3.legend()

    # Plot phase for the observed signal
    ax4 = axes[1, 1]
    ax4.set_title('Observed Signal (Phase)')
    c4 = ax4.plot(np.angle(dataset[0][:, 0]), label='Phase')
    ax4.set_xlabel('Antennas')
    ax4.set_ylabel('Phase (radians)')
    ax4.legend()

    plt.tight_layout()

    # Create an Animation object for iterating through samples
    ani = FuncAnimation(fig, update_data, frames=num_samples, fargs=(dataset, axes), interval=1000, repeat=False)

    plt.show()

def update_data(frame, dataset, axes):
    # Update the data for each subplot based on the current frame
    array_response_transposed = dataset[frame].T

    axes[0, 0].images[0].set_array(np.real(array_response_transposed))
    axes[0, 1].images[0].set_array(np.angle(array_response_transposed))

    axes[1, 0].lines[0].set_ydata(np.real(dataset[frame][:, 0]))
    axes[1, 0].lines[1].set_ydata(np.imag(dataset[frame][:, 0]))

    axes[1, 1].lines[0].set_ydata(np.angle(dataset[frame][:, 0]))

# Example usage
num_samples = 10
num_antennas = 4
num_snapshots = 10
num_sources = [1]  # Number of sources for each sample
coherent = False

# Assuming other parameters are defined
dataset, _ = create_broadband_dataset("temp_dataset", num_samples, num_sources, coherent, save=False)

# Visualize the dataset
visualize_dataset_2d(dataset)
