import os
import logging
import librosa
from joblib import load
from sklearn.mixture import GaussianMixture

def load_gmm_model(model_path: str) -> GaussianMixture:
    if not os.path.exists(model_path):
        logging.error(f"Model path {model_path} does not exist.")
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    try:
        return load(model_path)
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

def extract_mfcc(audio_path: str, sr: float | None = 16000, n_mfcc: int = 13, frame_length: int = 2048, hop_length: int = 512):
    if not os.path.exists(audio_path):
        logging.error(f"Model path {audio_path} does not exist.")
        raise FileNotFoundError(f"Model path {audio_path} does not exist.")
    
    y, sr = librosa.load(audio_path, sr=sr)
    
    if y is None:
        logging.error(f"Failed to load audio from {audio_path}")
        raise ValueError("Failed to load audio")
                       
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
    
    # Apply CMVN
    cmvn_mfccs = (mfccs - np.mean(mfccs, axis=0)) / np.std(mfccs, axis=0)
    
    return cmvn_mfccs.T

from sklearn.mixture import GaussianMixture

import numpy as np

def map_adaptation(gmm: GaussianMixture, mfcc_features: np.ndarray):
    if mfcc_features.shape[1] != gmm.means_.shape[1]:
        logging.error("Number of MFCC features does not match")
        raise ValueError("Number of MFCC features does not match the GMM model")
    posteriors = gmm.predict_proba(mfcc_features)
    adapted_means = np.dot(posteriors.T, mfcc_features) / (posteriors.sum(axis=0)[:, None] + 1e-6)
    adapted_covariances = []
    for k in range(gmm.n_components):
        diff = mfcc_features - gmm.means_[k]
        weighted_diff = diff * posteriors[:, [k]]
        cov = np.dot(weighted_diff.T, diff) / (posteriors[:, [k]].sum() + 1e-6)
        adapted_covariances.append(cov)
    
    gmm.means_ = adapted_means
    gmm.covariances_ = np.array(adapted_covariances)
    return gmm

def classify_audio(mfcc_features: np.ndarray, gmm_0: GaussianMixture, gmm_1: GaussianMixture) -> np.bool:
    if mfcc_features.shape[1] != gmm_1.means_.shape[1]:
        raise ValueError("Number of MFCC features does not match the GMM model for speaker 1")
    if mfcc_features.shape[1] != gmm_0.means_.shape[1]:
        raise ValueError("Number of MFCC features does not match the GMM model for speaker 0")

    speech_log_likelihood = gmm_1.score_samples(mfcc_features)
    noise_log_likelihood = gmm_0.score_samples(mfcc_features)
    
    if np.mean(speech_log_likelihood) > np.mean(noise_log_likelihood):
        return np.bool(1)
    else:
        return np.bool(0)

from collections import deque

class AudioBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque()

    def add_frame(self, frame):
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()  # Remove the oldest frame
        self.buffer.append(frame)

    def get_buffer(self):
        return list(self.buffer)

import argparse

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Classify audio between two trained Gaussian Mixture Models")
    parser.add_argument("audio_path", help = "Path to the input audio file")
    parser.add_argument("--gmm_0_path", required = True, help = "Path to the GMM model for speaker 0")
    parser.add_argument("--gmm_1_path", required = True, help = "Path to the GMM model for speaker 1")

    args = parser.parse_args()
    
    # Import GMMs for speakers 0 and 1 from provided arguments
    gmm0 = load_gmm_model(args.gmm_0_path)
    gmm1 = load_gmm_model(args.gmm_1_path)
    
    # Extract MFCC features from the provided audio file
    mfcc_features = extract_mfcc(args.audio_path)
    
    # Initialize speaker verification output
    output = np.zeros(len(mfcc_features), dtype=np.bool_)
    
    # Initialize audio buffer
    buffer_size: int = 5
    audio_buffer = AudioBuffer(max_size = buffer_size)
    
    # Loop through the audio
    for (i, frame) in enumerate(mfcc_features):
        audio_buffer.add_frame(frame)
        output[i] = classify_audio(mfcc_features=audio_buffer.get_buffer, gmm_0=gmm0, gmm_1=gmm1)
        
    # Print the speaker verification results
    print(f"Speaker Verification Results: {output}")

if __name__ == "__main__":
    main()