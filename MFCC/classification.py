import os
import logging
import librosa
from sklearn.mixture import GaussianMixture
import numpy as np
from collections import deque
import argparse
import matplotlib.pyplot as plt
from MFCC.common import load_gmm_model

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

def classify_audio(mfcc_features: np.ndarray, gmm_0: GaussianMixture, gmm_1: GaussianMixture) -> np.bool:
    feature_shape = mfcc_features.shape[1]
    gmm_0_shape = np.asarray(gmm_0.means_).shape
    gmm_1_shape = np.asarray(gmm_1.means_).shape

    if  feature_shape != gmm_1_shape[1]:
        raise ValueError("Number of MFCC features does not match the GMM model for speaker 1")
    if feature_shape != gmm_0_shape[1]:
        raise ValueError("Number of MFCC features does not match the GMM model for speaker 0")

    gmm_1_log_likelihood = gmm_1.score(mfcc_features)
    gmm_0_log_likelihood = gmm_0.score(mfcc_features)

    if gmm_1_log_likelihood > gmm_0_log_likelihood:
        return np.bool(1)
    else:
        return np.bool(0)

class FrameBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque()

    def add_frame(self, frame):
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()  # Remove the oldest frame
        self.buffer.append(frame)

    def get_buffer(self) -> np.ndarray:
        return np.array(list(self.buffer))

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Classify audio between two trained Gaussian Mixture Models")
    parser.add_argument("audio_path", help = "Path to the input audio file")
    parser.add_argument("--gmm_0_path", required = True, help = "Path to the GMM model for speaker 0")
    parser.add_argument("--gmm_1_path", required = True, help = "Path to the GMM model for speaker 1")
    parser.add_argument("-return_single", help= "Return only a single boolean for the entire audio sequence", action="store_true")
    parser.add_argument("-v", help="Enable verbose logging", action="store_true")

    args = parser.parse_args()

    # Import GMMs for speakers 0 and 1 from provided arguments
    gmm0 = load_gmm_model(args.gmm_0_path)
    gmm1 = load_gmm_model(args.gmm_1_path)

    # Extract MFCC features from the provided audio file
    mfcc_features = extract_mfcc(args.audio_path)

    if not args.return_single:
    
        # Initialize speaker verification output
        output = np.zeros(len(mfcc_features), dtype=np.bool_)

        # Initialize audio buffer
        buffer_size: int = 2
        audio_buffer = FrameBuffer(max_size = buffer_size)

        # Loop through the audio
        for (i, frame) in enumerate(mfcc_features):
            audio_buffer.add_frame(frame)
            current_buffer = audio_buffer.get_buffer()
            output[i] = classify_audio(current_buffer, gmm_0=gmm0, gmm_1=gmm1)

        # Plot the results
        plt.figure(figsize=(12, 4))
        plt.plot(output, drawstyle='steps-post')
        plt.title('Speaker Identification Over Time')
        plt.xlabel('Frame Index (Time)')
        plt.ylabel('Speaker Detected (0 = No, 1 = Yes)')
        plt.grid(True)
        plt.show()
        
    else:
        print(classify_audio(mfcc_features=mfcc_features, gmm_0=gmm0,gmm_1=gmm1))

if __name__ == "__main__":
    main()