import logging
import numpy as np
from sklearn.mixture import GaussianMixture
import argparse
from MFCC.classification import load_gmm_model, extract_mfcc
from MFCC.train import save_gmm_model
import datetime

def map_adaptation(gmm: GaussianMixture, mfcc_features: np.ndarray) -> GaussianMixture:
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

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Adapt a UBM to a specific speaker")
    parser.add_argument("ubm_path", help = "Path to the universal background model")
    parser.add_argument("--audio_path", required = True, help = "Path to the input audio file")
    parser.add_argument("--save_path", required = True, help="Path to the directory to save speaker GMM")
    
    args = parser.parse_args()
    
    ubmPath: str = args.ubm_path
    audioPath: str = args.audio_path
    baseSavePath: str = args.save_path
    
    ubm: GaussianMixture = load_gmm_model(model_path=ubmPath)
    
    mfccs = extract_mfcc(audio_path=audioPath)
    
    gmm: GaussianMixture = map_adaptation(ubm,mfcc_features=mfccs)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    savePath = baseSavePath + f"/gmm_{timestamp}.pkl"
    
    save_gmm_model(gmm=gmm, model_path=savePath)
    
    
if __name__ == "__main__":
    main()