import librosa

def extract(audio_path: str, sr: int = 16000, n_mfcc: int = 13, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, frame_length=frame_length, hop_length=hop_length)
    return mfccs.T

from sklearn.mixture import GaussianMixture

def train_gmm(features, n_components: int = 6) -> GaussianMixture:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(features)
    return gmm

import numpy as np

def classify_audio(gmm_1: GaussianMixture, gmm_0: GaussianMixture, mfcc_features: np.ndarray) -> np.bool:
    speech_log_likelihood = gmm_1.score_samples(mfcc_features)
    noise_log_likelihood = gmm_0.score_samples(mfcc_features)
    
    if np.mean(speech_log_likelihood) > np.mean(noise_log_likelihood):
        return np.bool(1)
    else:
        return np.bool(0)