import argparse
import logging
import os
import pickle
import sys
import warnings

import librosa.display
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from librosa import load
from librosa.feature import mfcc, melspectrogram
from scipy.signal import medfilt
from utilities import *
import yaml

warnings.filterwarnings("ignore")


def process_audio_files(config: dict) -> None:
    """
    Process audio files by calculating MFCC and mel-spectrogram features,
    applying filters, and saving processed data to pickle files.

    Args:
        config (dict): Configuration dictionary with necessary parameters.
    """

    SR = config["sampling_rate"]
    L = config["frame_length"]
    U = config["frame_step"]

    output_directory = config["output_directory"]
    os.makedirs(output_directory, exist_ok=True)

    # Convert from seconds to samples
    n_frame_length = round(L * SR)
    n_frame_step = round(U * SR)

    audio_clips_mfcc = []
    audio_clips_mel_spectrogram = []

    # Count files
    total_files = 0
    for root, _, files in os.walk(config["data_path"], topdown=False):
        for name in files:
            total_files += 1

    # Traverse directory with audio files
    for root, _, files in os.walk(config["data_path"], topdown=False):
        for i, name in tqdm(enumerate(files), total=total_files):

            logging.info(f"Processing file {i+1}: {name}")
            file = os.path.join(root, name)
            filename = os.path.splitext(os.path.basename(name))[0]

            librosa_plots_path = os.path.join(output_directory, "librosa_plots")
            os.makedirs(librosa_plots_path, exist_ok=True)

            threshold_plots_path = os.path.join(
                output_directory, "custom_threshold_plots"
            )
            os.makedirs(threshold_plots_path, exist_ok=True)

            try:
                # Load audio file
                y, _ = load(file, sr=SR)

                # Apply FIR filter and remove DC component
                y = fir_filter(y, SR, 200, 4000, 37, np.float32)
                y = y - np.mean(y)

                # Detect voiced segments
                voiced_parts_lib, _ = voiced_unvoiced_detection_librosa(
                    y, n_frame_length, SR, os.path.join(librosa_plots_path, filename), save_plot=False
                )
                voiced_parts_thr, _ = voiced_unvoiced_detection_threshold(
                    y,
                    n_frame_length,
                    n_frame_step,
                    SR,
                    os.path.join(threshold_plots_path, filename),
                    save_plot=False
                )

                # Compute mel-spectrogram and MFCC features for the full clip and voiced segments
                audio_clips_mel_spectrogram.append(
                    [
                        melspectrogram(y=y, sr=SR).mean(axis=1).tolist(),
                        melspectrogram(
                            y=y[voiced_parts_lib[0] : voiced_parts_lib[1]], sr=SR
                        )
                        .mean(axis=1)
                        .tolist(),
                        melspectrogram(
                            y=y[voiced_parts_thr[0] : voiced_parts_thr[1]], sr=SR
                        )
                        .mean(axis=1)
                        .tolist(),
                        int(file[19]),
                    ]
                )
                audio_clips_mfcc.append(
                    [
                        mfcc(y=y, sr=SR).mean(axis=1).tolist(),
                        mfcc(
                            y=y[voiced_parts_lib[0] : voiced_parts_lib[1]],
                            sr=SR,
                            n_mels=128,
                        )
                        .mean(axis=1)
                        .tolist(),
                        mfcc(
                            y=y[voiced_parts_thr[0] : voiced_parts_thr[1]],
                            sr=SR,
                            n_mels=128,
                        )
                        .mean(axis=1)
                        .tolist(),
                        int(file[19]),
                    ]
                )
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")
                continue
    # Save the processed audio features to pickle files
    try:
        with open(os.path.join(output_directory, "audio_clips_mfcc"), "wb") as fp:
            pickle.dump(audio_clips_mfcc, fp)
        with open(
            os.path.join(output_directory, "audio_clips_mel_spectrogram"), "wb"
        ) as fp:
            pickle.dump(audio_clips_mel_spectrogram, fp)
        logging.info("Successfully saved processed audio clips to pickle files.")
    except Exception as e:
        logging.error(f"Error saving data to pickle files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio processing script for MFCC and Mel-spectrogram extraction."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load configuration file
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file {args.config} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {args.config}: {e}")
        sys.exit(1)

    # Run the main processing function
    process_audio_files(config)
