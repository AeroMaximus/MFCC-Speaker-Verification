import os
import logging
import librosa
from joblib import dump, load
from sklearn.mixture import GaussianMixture

def extract_mfcc(audio_path: str, sr: float | None = 16000, n_mfcc: int = 13, frame_length: int = 2048, hop_length: int = 512, n_deltas: int = 2):
    if not os.path.exists(audio_path):
        logging.error(f"Audio path {audio_path} does not exist.")
        raise FileNotFoundError(f"Audio path {audio_path} does not exist.")

    y, sr = librosa.load(audio_path, sr=sr)

    if y is None:
        logging.error(f"Failed to load audio from {audio_path}")
        raise ValueError("Failed to load audio")

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)

    # Apply CMVN
    mean = np.mean(mfccs, axis=0)
    std = np.std(mfccs, axis=0)
    cmvn_mfccs = (mfccs - mean) / (std + 1e-6)

    # Set the first coefficient to zero to eliminate engergy dependence
    cmvn_mfccs[0, :] = 0

    # Calculate deltas and delta-deltas if requested
    if n_deltas == 1 or n_deltas == 2:
        deltas = librosa.feature.delta(cmvn_mfccs)
        cmvn_mfccs = np.vstack((cmvn_mfccs, deltas))

        if n_deltas == 2:
            delta_deltas = librosa.feature.delta(deltas)
            cmvn_mfccs = np.vstack((cmvn_mfccs, delta_deltas))
    else:
        logging.error(f"{n_deltas} is an invalid number of deltas, only 0, 1, or 2 are allowed.")
        raise ValueError("Number of deltas is an invalid number")

    return cmvn_mfccs.T

def find_files_with_extension(directory, extensions):
    """
    Find all files in a directory with given extensions.

    :param directory: Path to the directory to search.
    :param extensions: A single file extension or a list of file extensions.
    :return: List of paths to files with matching extensions.
    """
    if isinstance(extensions, str):
        extensions = [extensions]

    matching_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    matching_files.append(os.path.join(root, file))
                    break

    return matching_files

import numpy as np
import hashlib

def array_to_hash(array):
    # Convert the array to a string
    array_str = str(array)
    
    # Create a sha256 hash object
    hash_object = hashlib.sha256()
    
    # Update the hash object with the bytes of the array string
    hash_object.update(array_str.encode('utf-8'))
    
    # Get the hexadecimal representation of the hash
    full_hash = hash_object.hexdigest()
    
    return full_hash

def npy_to_hash(npy_path):
    # Load the array from the .npy file
    array = np.load(npy_path)
    
    # Convert the array to a string representation
    array_str = str(array)
    
    # Create a sha256 hash object
    hash_object = hashlib.sha256()
    
    # Update the hash object with the bytes of the array string
    hash_object.update(array_str.encode('utf-8'))
    
    # Get the hexadecimal representation of the hash
    full_hash = hash_object.hexdigest()
    
    return full_hash

def save_gmm_model(gmm: GaussianMixture, model_path: str):
    try:
        # Convert relative path to absolute path if necessary
        abs_model_path = os.path.abspath(model_path)
        dump(gmm, abs_model_path)
    except Exception as e:
        logging.error(f"Failed to save model to {model_path}: {str(e)}")
        raise
    
def load_gmm_model(model_path: str) -> GaussianMixture:
    if not os.path.exists(model_path):
        logging.error(f"Model path {model_path} does not exist.")
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    try:
        return load(model_path)
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {str(e)}")
        raise