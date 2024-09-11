import matplotlib.pyplot as plt
from librosa import (
    get_duration,
    frames_to_time,
    frames_to_samples,
    onset,
    time_to_samples,
    samples_to_time,
    amplitude_to_db,
    stft,
)
import numpy as np
from scipy.signal import medfilt
import sounddevice as sd
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import logging

warnings.filterwarnings("ignore")


def fir_filter(
    samples: np.ndarray, fs: int, fL: int, fH: int, N: int, outputType: type
) -> np.ndarray:
    """
    Apply a high-pass and low-pass FIR filter to audio samples.

    Args:
        samples (np.ndarray): The input audio signal.
        fs (int): The sampling frequency.
        fL (int): The low cutoff frequency.
        fH (int): The high cutoff frequency.
        N (int): The filter order.
        outputType (type): The desired output data type for the filtered signal.

    Returns:
        np.ndarray: The filtered audio signal.
    """
    logging.info("Applying FIR filter: High-pass and Low-pass.")

    # Normalize cutoff frequencies
    fH = fH / fs
    fL = fL / fs

    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = np.sinc(2 * fH * (np.arange(N) - (N - 1) / 2.0))
    hlpf *= np.blackman(N)  # Apply windowing function
    hlpf /= np.sum(hlpf)  # Normalize filter

    # Compute a high-pass filter with cutoff frequency fL.
    hhpf = np.sinc(2 * fL * (np.arange(N) - (N - 1) / 2.0))
    hhpf *= np.blackman(N)
    hhpf /= np.sum(hhpf)

    # Create a high-pass filter from the low-pass filter through spectral inversion.
    hhpf = -hhpf
    hhpf[int((N - 1) / 2)] += 1

    # Convolve both filters.
    h = np.convolve(hlpf, hhpf)
    logging.debug("Filter coefficients computed.")

    # Apply the filter to the audio signal.
    s = np.convolve(samples, h).astype(outputType)
    logging.info("FIR filtering completed.")

    return s


def voiced_unvoiced_detection_librosa(
    y: np.ndarray, frame_length: int, s: int, filename: str, save_plot: bool
) -> tuple:
    """
    Detect voiced and unvoiced parts of an audio signal using Librosa's onset detection.

    Args:
       y (np.ndarray): The input audio signal.
       frame_length (int): The frame length in samples.
       s (int): The sampling rate.
       filename (str): Path to which the plot will be saved.
       save_plot (bool): Boolean that indicates whether to save the plot as a file.

    Returns:
       tuple: A tuple containing:
          - onset_samples (np.ndarray): The start and end sample indices of the voiced clips.
          - digits (int): The number of detected voiced segments (onsets).
    """
    logging.info("Starting voiced/unvoiced detection using Librosa.")

    # Get duration of audio clip in seconds
    dur = get_duration(y=y, sr=s)

    # Reverse the audio clip for onset detection at the end
    y_reversed = y[::-1]

    # Detect the onset frames for the original and reversed audio
    onset_frames = onset.onset_detect(
        y=y, sr=s, hop_length=frame_length, backtrack=True
    )
    onset_reversed_frames = onset.onset_detect(
        y=y_reversed, sr=s, hop_length=frame_length, backtrack=True
    )

    # Convert onset frames to time in seconds and samples
    onset_time = frames_to_time(onset_frames, sr=s, hop_length=frame_length)
    onset_samples = frames_to_samples(onset_frames, hop_length=frame_length)
    onset_reversed_time = frames_to_time(
        onset_reversed_frames, sr=s, hop_length=frame_length
    )

    # Adjust the reversed onset times to correspond to the original audio
    for i in range(len(onset_reversed_time)):
        onset_reversed_time[i] = dur - onset_reversed_time[i]
    logging.debug("Reversed onset times adjusted.")

    # Sort the end of each voiced segment (original audio was reversed)
    onset_reversed_time = sorted(onset_reversed_time)

    # Merge and clean up the onset times
    # Remove redundant ends if they are too close together (< 1 sec)
    i = 0
    while i < len(onset_reversed_time) - 1:
        if onset_reversed_time[i + 1] - onset_reversed_time[i] < 1:
            onset_reversed_time = np.delete(onset_reversed_time, i)
            i -= 1
        i += 1

    # Do the same for the start times
    i = 0
    while i < len(onset_time) - 1:
        if onset_time[i + 1] - onset_time[i] < 1:
            onset_time = np.delete(onset_time, i + 1)
            i -= 1
        i += 1

    # Merge start and end times of each voiced segment
    merged_onset_time = sorted([*onset_time, *onset_reversed_time])

    # Convert merged onset times to sample indices
    onset_samples = time_to_samples(merged_onset_time, sr=s)

    # Create a binary voiced/unvoiced label array for the entire audio signal
    voiced_unvoiced_labels = [0] * len(y)
    for i in range(len(onset_samples)):
        if i % 2 == 0 and i + 1 < len(onset_samples):
            for j in range(onset_samples[i], onset_samples[i + 1] + 1):
                voiced_unvoiced_labels[j] = 1

    # Ensure the merged onset time array has an even number of elements
    if len(merged_onset_time) % 2 == 1:
        merged_onset_time.pop()

    # Number of voiced segments (onsets)
    digits = len(merged_onset_time) // 2
    logging.info(f"Detected {digits} voiced segments.")

    if save_plot:
        # Show the voiced and unvoiced parts with an outline
        plt.figure(figsize=(12, 8), dpi=100)
        plt.suptitle("Voiced parts using librosa's onset_detection function")
        plt.subplot(211)
        plt.title("Spectroscopy representation with voiced parts outlined")
        y_db = amplitude_to_db(abs(stft(y)),ref=np.max)
        librosa.display.specshow(data=y_db, sr=s, x_axis="time", y_axis='log')
        plt.vlines(merged_onset_time, 0, s, color="r")
        plt.subplot(212)
        plt.title("Waveplot representation with voiced parts outlined")
        plt.plot(voiced_unvoiced_labels, color="r", linestyle="-")
        plt.plot(y)
        plt.tight_layout()
        plt.savefig(filename)

    return onset_samples, round(digits)


# With overlapping windows
def voiced_unvoiced_detection_threshold(
    y: np.ndarray, n_frame_length: int, n_frame_step: int, s: int, filename: str, save_plot: bool
) -> tuple:
    """
    Detect voiced and unvoiced parts of an audio signal using energy and zero-crossing rate (ZCR) thresholds.

    Args:
       y (np.ndarray): The input audio signal.
       n_frame_length (int): The frame length in samples.
       n_frame_step (int): The step size between consecutive frames in samples.
       s (int): The sampling rate.
       filename (str): Path to which the plot will be saved.
       save_plot (bool): Boolean that indicates whether to save the plot as a file.
       

    Returns:
       tuple: A tuple containing:
          - onset_samples (list): The start sample indices of voiced clips.
          - digits (int): The number of detected voiced segments (onsets).
    """
    logging.info("Starting voiced/unvoiced detection using energy and ZCR thresholds.")

    # Audio length in # of samples
    D = len(y)

    # Create Hamming window for smoothing each frame
    win = np.hamming(n_frame_length)

    # Initialize lists for energy and zero-crossing rate (ZCR)
    energy = []
    zcr = []

    # Iterate over frames with overlapping windows
    for frame in range(0, (D - (n_frame_length - 1)), n_frame_step):
        start = frame
        end = (frame + n_frame_length - 1) + 1

        # Apply Hamming window to the frame
        window = [y[start:end][i] * win[i] for i in range(len(win))]

        # Compute the energy and zero-crossing rate of the frame
        energy.append((1 / len(window)) * sum(i * i for i in window))
        zcr.append(
            0.5
            * sum(
                abs(
                    np.sign(window[2 : len(window)])
                    - np.sign(window[1 : len(window) - 1])
                )
            )
        )

    # Set thresholds based on mean and standard deviation of energy and ZCR
    energy_threshold = np.mean(energy) / 5
    zcr_threshold = 3 * np.mean(zcr) - 0.2 * np.std(zcr, ddof=1)

    # Initialize labels for voiced/unvoiced frames and auxiliary arrays for plotting
    voiced_unvoiced_labels = []
    plot_energy = []
    plot_zcr = []

    # Determine whether each frame is voiced or unvoiced based on energy and ZCR thresholds
    for i in range(len(energy)):
        if i == 0:
            plot_energy.extend([energy[i]] * n_frame_length)
            plot_zcr.extend([zcr[i]] * n_frame_length)
            if energy[i] <= energy_threshold:
                voiced_unvoiced_labels.extend([0] * n_frame_length)
            else:
                if zcr[i] <= zcr_threshold:
                    voiced_unvoiced_labels.extend([1] * n_frame_length)
                else:
                    voiced_unvoiced_labels.extend([0] * n_frame_length)
        else:
            plot_energy.extend([energy[i]] * n_frame_step)
            plot_zcr.extend([zcr[i]] * n_frame_step)
            if energy[i] <= energy_threshold:
                voiced_unvoiced_labels.extend([0] * n_frame_step)
            else:
                if zcr[i] <= zcr_threshold:
                    voiced_unvoiced_labels.extend([1] * n_frame_step)
                else:
                    voiced_unvoiced_labels.extend([0] * n_frame_step)

    # Apply a median filter to smooth out noise in the labels
    voiced_unvoiced_labels = medfilt(voiced_unvoiced_labels, 1001)

    # Identify onset times and sample indices where voiced segments start or end
    onset_time = []
    onset_samples = []
    for i in range(1, len(voiced_unvoiced_labels) - 1):
        if (voiced_unvoiced_labels[i] == 1 and voiced_unvoiced_labels[i - 1] == 0) or (
            voiced_unvoiced_labels[i] == 1 and voiced_unvoiced_labels[i + 1] == 0
        ):
            onset_time.append(samples_to_time(samples=i, sr=s))
            onset_samples.append(i)

    # Number of voiced segments detected
    digits = len(onset_samples) // 2
    logging.info(f"Detected {digits} voiced segments using thresholding.")

    if save_plot:
        # Show the voiced and unvoiced parts with an outline
        plt.figure(figsize=(12, 8), dpi=100)
        plt.suptitle("Voiced parts using energy/zcr thresholds")
        plt.subplot(221)
        plt.title("Spectroscopy representation with voiced parts outlined")
        y_db = amplitude_to_db(abs(stft(y)),ref=np.max)
        librosa.display.specshow(y_db, sr=s, x_axis="time", y_axis="log")
        plt.vlines(onset_time, 0, s, color="r")
        plt.subplot(223)
        plt.title("Waveplot representation with voiced parts outlined")
        plt.plot(voiced_unvoiced_labels, color="r", linestyle="-")
        plt.plot(y)
        plt.subplot(222)
        plt.title("Energy of signal")
        plt.plot(plot_energy, color="g", linestyle="-")
        plt.subplot(224)
        plt.title("Zero crossing rate of signal")
        plt.plot(plot_zcr, color="b", linestyle="-")
        plt.tight_layout()
        plt.savefig(filename)

    return onset_samples, round(digits)
