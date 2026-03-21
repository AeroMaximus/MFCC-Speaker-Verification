import os
import logging
from joblib import dump, load
from sklearn.mixture import GaussianMixture

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
        logging.error(f"Failed to save model to {abs_model_path}: {str(e)}")
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