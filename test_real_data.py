import os
import sys
import argparse
import yaml
import warnings
import logging
import numpy as np
import sounddevice as sd
from librosa.feature import melspectrogram
from librosa import load
from scipy.io.wavfile import write, read
import tensorflow as tf
from tensorflow.keras import layers
from utilities import fir_filter, voiced_unvoiced_detection_librosa, voiced_unvoiced_detection_threshold
from tensorflow import keras
import time

# Suppress warnings and set TensorFlow logging level
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def record_audio(duration: int, sample_rate: int) -> np.ndarray:
    """
    Record audio for a given duration.

    Args:
        duration (int): Duration of the recording in seconds.
        sample_rate (int): Sampling rate of the audio.

    Returns:
        np.ndarray: Recorded audio data.
    """
    logging.info(f"Recording for {duration} seconds.")
    data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    if data.ndim==1:
        return data
    else:
        return data[:, 0]

def load_audio_file(filepath: str) -> np.ndarray:
    """
    Load an audio file.

    Args:
        filepath (str): Path to the audio file.

    Returns:
        np.ndarray: Loaded audio data.
    """
    try:
        logging.info(f"Loading audio file: {filepath}")
        _, data = read(filepath)
        if data.ndim==1:
            return data
        else:
            return data[:, 0]
    except FileNotFoundError:
        logging.error(f"File {filepath} not found.")
        return None


def load_models(output_directory) -> tuple[keras.Model, keras.Model, keras.Model]:
    """
    Load pre-trained Keras models for digit classification.

    Returns:
        tuple: Three pre-trained Keras models.
    """
    logging.info("Loading pre-trained models.")
    model_lib = keras.models.load_model(os.path.join(output_directory, 'models', 'model_voiced_clips_librosa_mel_spectrogram.keras'))
    model_orig = keras.models.load_model(os.path.join(output_directory, 'models', 'model_original_clips_mel_spectrogram.keras'))
    model_thr = keras.models.load_model( os.path.join(output_directory, 'models', 'model_voiced_clips_threshold_mel_spectrogram.keras'))
    return model_lib, model_orig, model_thr

def predict_digits(model: keras.Model, spectrograms: list, digits_num: int) -> list:
    """
    Predict digits based on mel spectrograms using a given model.

    Args:
        model (keras.Model): Pre-trained Keras model.
        spectrograms (list): List of mel spectrograms.
        digits_num (int): Number of digits to predict.

    Returns:
        list: Predicted digits.
    """
    logging.info(f"Predicting {digits_num} digits.")
    predictions = []
    for i in range(digits_num):
        pred = model.predict(np.array(spectrograms[i]))
        predictions.append(np.where(pred[0] == 1)[0])
    return predictions

def main(config) -> None:
    """
    Main loop to handle user input for file selection, recording, and digit prediction.
    """
    output_directory = config["output_directory"]
    SECONDS = config["duration"]
    SR = config["sampling_rate"]
    L = config["frame_length"]
    U = config["frame_step"]
    n_frame_length, n_frame_step = round(L * SR), round(U * SR)  # Convert to samples
    
    model_lib, model_orig, model_thr = load_models(output_directory)
    
    while True:
        logging.info('Waiting for user input (F for file, R to record, E to exit).')
        choice = input('Write F for file selection, R to record, or E to exit: ').lower()

        if choice == 'f':
            filepath = input('Enter the file path: ')
            filename = os.path.basename(filepath)
            data = load_audio_file(filepath)
            if data is None:
                continue

        elif choice == 'r':
            logging.info('Recording audio...')
            print(f'Recording for {SECONDS} seconds. Please speak digits clearly with pauses.')
            data = record_audio(SECONDS, SR)
            filename = f'recorded_digits_{time.time()}.wav'
            os.makedirs(os.path.join(output_directory, 'recorded_audios'), exist_ok=True)
            write(os.path.join(output_directory, 'recorded_audios', filename), SR, data)
            

        elif choice == 'e':
            logging.info('Exiting program.')
            break

        else:
            logging.warning('Invalid input, please try again.')
            continue

        # Apply FIR filter and remove DC component
        data = fir_filter(data, SR, 200, 4000, 37, np.float32)
        processed_data = data - np.mean(data)
        
        os.makedirs(os.path.join(output_directory, 'prediction_plots'), exist_ok=True)
        voiced_parts_lib, digits_lib_num = voiced_unvoiced_detection_librosa(
            processed_data, n_frame_length, SR, os.path.join(output_directory, 'prediction_plots', f'librosa_{filename}.png'), save_plot=True)
        voiced_parts_thr, digits_thr_num = voiced_unvoiced_detection_threshold(
            processed_data, n_frame_length, n_frame_step, SR, os.path.join(output_directory, 'prediction_plots', f'threshold_{filename}.png'), save_plot=True)

        # Extract mel spectrograms for both methods
        digits_lib = []
        for i in range(digits_lib_num):
            digits_lib.append([melspectrogram(y=processed_data[voiced_parts_lib[2*i]:voiced_parts_lib[2*i+1]], sr=10000).mean(axis=1).tolist()])

        digits_thr = []
        for i in range(digits_thr_num):
            digits_thr.append([melspectrogram(y=processed_data[voiced_parts_thr[2*i]:voiced_parts_thr[2*i+1]], sr=10000).mean(axis=1).tolist()])
            

        # Predict using the librosa preprocessed model
        logging.info(f'Based on librosa, {digits_lib_num} digits detected.')
        predictions_lib = predict_digits(model_lib, digits_lib, digits_lib_num)
        for i, pred in enumerate(predictions_lib):
            logging.info(f'Digit {i + 1} predicted as: {pred}')

        # Predict using the original clips model (with librosa preprocessing)
        logging.info('Predicting digits using the original clips model (with librosa preprocessing).')
        predictions_orig = predict_digits(model_orig, digits_lib, digits_lib_num)
        for i, pred in enumerate(predictions_orig):
            logging.info(f'Digit {i + 1} predicted as: {pred}')

        # Predict using the threshold preprocessed model
        logging.info(f'Based on thresholds, {digits_thr_num} digits detected.')
        predictions_thr = predict_digits(model_thr, digits_thr, digits_thr_num)
        for i, pred in enumerate(predictions_thr):
            logging.info(f'Digit {i + 1} predicted as: {pred}')

        # Predict using the original clips model (with threshold preprocessing)
        logging.info('Predicting digits using the original clips model (with threshold preprocessing).')
        predictions_orig_thr = predict_digits(model_orig, digits_thr, digits_thr_num)
        for i, pred in enumerate(predictions_orig_thr):
            logging.info(f'Digit {i + 1} predicted as: {pred}')



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Audio processing script for MFCC and Mel-spectrogram extraction.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Set up logging configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    
    main(config)
