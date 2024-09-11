import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import seaborn as sn
import pandas as pd
import warnings
import logging
import argparse
import yaml
import os
import sys
from librosa import load

warnings.filterwarnings("ignore")

# Check for GPU availability and log a message
if tf.test.gpu_device_name():
    logging.info(f'Default GPU Device: {tf.test.gpu_device_name()}')
else:
    logging.warning("Please install the GPU version of TensorFlow")

def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load audio clips and labels from a pickle file.

    Args:
        filepath (str): Path to the pickle file containing audio clips.

    Returns:
        tuple: Contains arrays of original clips, voiced clips (librosa), 
               voiced clips (threshold-based), and labels.
    """
    try:
        with open(filepath, 'rb') as handle:
            audio_clips = pickle.load(handle)
        original_clips, voiced_clips_librosa, voiced_clips_threshold, labels = [], [], [], []
        for audio in audio_clips:
            original_clips.append(audio[0])
            voiced_clips_librosa.append(audio[1])
            voiced_clips_threshold.append(audio[2])
            labels.append(audio[3])
        logging.info("Data successfully loaded from file.")
        return (np.array(original_clips), np.array(voiced_clips_librosa), 
                np.array(voiced_clips_threshold), np.array(labels))
    except FileNotFoundError:
        logging.error(f"Pickle file not found at {filepath}.")
        sys.exit()
        return None

def plot_extra(data_path: str, output_directory: str, sr: int) -> None:
    """
    Generate and save various plots for augmented audio signals and mel spectrograms.

    Args:
        data_path (str): Path to the directory containing audio files.
        output_directory (str): Directory where plots will be saved.
        sr (int): Sampling rate for the audio files.
    """
    # Load a sample audio file
    y, _ = load(os.path.join(data_path, '01', '0_01_0.wav'), sr=sr)

    # Create output directories for plots if they don't exist
    plot_path = os.path.join(output_directory, 'plots')
    os.makedirs(plot_path, exist_ok=True)

    # Define the augmentation pipeline
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
    ])
    augmented_samples = augment(samples=y, sample_rate=sr)

    # Plot augmented and original audio signals
    plt.figure(1)
    plt.subplot(211)
    plt.title('Augmented audio signal')
    plt.plot(augmented_samples)
    plt.xlabel('Time')

    plt.subplot(212)
    plt.title('Original audio signal')
    plt.plot(y)
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'augmented_signal.png'))
    plt.close()

    # Plot mel-spectrograms and MFCCs for digits
    fig = plt.figure(figsize=(25, 14), dpi=70)
    subfigs = fig.subfigures(2, 5)
    for digit, subfig in enumerate(subfigs.flat):
        subfig.suptitle(f'Digit {digit}')
        y, _ = load(os.path.join(data_path, '01', f'{digit}_01_0.wav'), sr=sr)
        axs = subfig.subplots(2, 1)
        for idx, ax in enumerate(axs.flat):
            if idx == 1:
                ax.set_title(f'MFCC of digit {digit}', fontsize='small')
                img = librosa.display.specshow(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), x_axis='time', ax=ax)
                fig.colorbar(img, ax=ax)
            else:
                ax.set_title(f'Mel spectrogram of digit {digit}', fontsize='small')
                img = librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max),
                                               x_axis='time', y_axis='mel', ax=ax)
                fig.colorbar(img, ax=ax)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(plot_path, 'digits.png'))
    plt.close()

def build_model(input_shape: int) -> keras.Model:
    """
    Build and compile a simple neural network model for classification.

    Args:
        input_shape (int): Shape of the input data.

    Returns:
        keras.Model: Compiled Keras model.
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    model.add(layers.Dense(input_shape, activation="relu"))
    model.add(layers.Dense(input_shape // 2, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    logging.info("Model built and compiled.")
    return model

def plot_metrics(history: keras.callbacks.History, test_acc: float, test_loss: float, 
                 matrix: np.ndarray, class_rep: dict, title: str, filepath: str) -> None:
    """
    Plot training history and evaluation metrics.

    Args:
        history (keras.callbacks.History): Training history object.
        test_acc (float): Test accuracy value.
        test_loss (float): Test loss value.
        matrix (np.ndarray): Confusion matrix.
        class_rep (dict): Classification report dictionary.
        title (str): Title for the plots.
        filepath (str): Path to save the plot.
    """
    plt.figure(dpi=220)
    sn.set(font_scale=0.4)
    plt.suptitle(f'{title} / DFT coefficients')

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot loss
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot confusion matrix
    plt.subplot(223)
    plt.title(f'Test Accuracy: {round(test_acc * 100)}%')
    sn.heatmap(pd.DataFrame(matrix, range(10), range(10)), annot=True, annot_kws={"size": 5}, fmt="d")

    # Plot classification report
    plt.subplot(224)
    plt.title(f'Test Loss: {round(test_loss * 100)}%')
    sn.heatmap(pd.DataFrame(class_rep).iloc[:-1, :].T, annot=True, annot_kws={"size": 5})

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def train_and_evaluate(config: dict, filepath: str, feature_type: str) -> None:
    """
    Train and evaluate models on different types of preprocessed data.

    Args:
        config (dict): Configuration settings from the YAML file.
        filepath (str): Path to the data.
        feature_type (str): Type of feature being processed (e.g., 'mfcc', 'mel_spectrogram').
    """
    output_directory = config["output_directory"]
    data_path = config["data_path"]
    sr = config["sampling_rate"]

    # Load data from the specified filepath
    data = load_data(filepath)
    if data:
        original_clips, voiced_clips_librosa, voiced_clips_threshold, labels = data
        plt_titles = ['No preprocessing', 'Preprocessed using librosa', 'Preprocessed using thresholds']
        data_type_names = ['original_clips', 'voiced_clips_librosa', 'voiced_clips_threshold']
        data_type = [original_clips, voiced_clips_librosa, voiced_clips_threshold]

    plot_extra(data_path, output_directory, sr)

    # Train and evaluate the model for each type of data
    for index in range(len(data_type)):
        X_train, X_test, y_train, y_test = train_test_split(data_type[index], labels, test_size=0.3, random_state=42)
        X_train, y_train, X_test, y_test = map(np.array, [X_train, y_train, X_test, y_test])

        model = build_model(len(X_train[0]))
        logging.info(model.summary())

        logging.info(f"Training model for {plt_titles[index]} data...")

        model_dir = os.path.join(output_directory, 'models')
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_dir, f"model_{data_type_names[index]}_{feature_type}.keras")

        # Configure callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)

        # Train the model
        history = model.fit(X_train, y_train, batch_size=64, validation_split=0.1, epochs=150, 
                            callbacks=[early_stopping, model_checkpoint], verbose=2)

        # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        y_pred = model.predict(X_test)

        # Compute confusion matrix and classification report
        matrix = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
        class_rep = classification_report(y_test, np.argmax(y_pred, axis=1), output_dict=True)

        logging.info(f'Test accuracy: {round(test_acc * 100)}%')
        logging.info(f'Test loss: {test_loss}')
        logging.info(f'Model {data_type_names[index]} saved at {checkpoint_path}')

        plot_filepath = os.path.join(output_directory, 'plots')
        os.makedirs(plot_filepath, exist_ok=True)
        plot_metrics(history, test_acc, test_loss, matrix, class_rep, plt_titles[index], 
                     os.path.join(plot_filepath, f'model_{data_type_names[index]}_{feature_type}.png'))

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

    # Train and evaluate models using the specified configuration
    output_directory = config["output_directory"]

    mel_spectrogram_path = os.path.join(output_directory, 'audio_clips_mel_spectrogram')
    train_and_evaluate(config, mel_spectrogram_path, 'mel_spectrogram')
    logging.info('Model with Mel Spectrogram features finished training.')

    mfcc_path = os.path.join(output_directory, 'audio_clips_mfcc')
    train_and_evaluate(config, mfcc_path, 'mfcc')
    logging.info('Model with MFCC features finished training.')