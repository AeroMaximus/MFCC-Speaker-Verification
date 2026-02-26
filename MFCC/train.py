import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from sklearn.mixture import GaussianMixture
from classification import extract_mfcc
from common import find_files_with_extension

def train_gmm(features, n_components: int = 6, reg_covar=1e-6) -> GaussianMixture:
    gmm = GaussianMixture(n_components=n_components, random_state=0, reg_covar=reg_covar)
    gmm.fit(features)
        
    return gmm

from joblib import dump

def save_gmm_model(gmm: GaussianMixture, model_path: str):
    try:
        dump(gmm, model_path)
    except Exception as e:
        logging.error(f"Failed to save model to {model_path}: {str(e)}")
        raise
    
import argparse
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
   
def main():
    parser = argparse.ArgumentParser(description="GMM generator")
    parser.add_argument("audio_dir", type=str, help="Path to the audio directory to train on")
    parser.add_argument("--n_components", type=int, help="Number of Gaussian Mixture components", default=512)
    parser.add_argument("--n_mfcc", type=int, help="Number of MFCC features to extract", default=39)
    args = parser.parse_args()
    
    logging.info("\033[1;32mFinding Audio Files...\033[0m")
    
    audio_files = find_files_with_extension(args.audio_dir, extensions=[".wav", ".flac", ".mp3"])
    
    if audio_files:   
        logging.info("\033[1;32mFound the audio files, proceeding with extraction...\033[0m")
    else:
        logging.warning("No audio files found. Please check the directory and file extensions.")
        return
    
    with Pool() as p:
        extract_mfcc_with_n_mfcc = partial(extract_mfcc, n_mfcc=args.n_mfcc)
        mfcc_features = [feat for feat in tqdm(p.imap(extract_mfcc_with_n_mfcc, audio_files), total=len(audio_files)) if feat is not None]
           
    if not mfcc_features:
        logging.error("No valid features extracted. Please check the audio files and extraction process.")
        return

    # Reshape so it's a 2D array
    mfcc_features = np.vstack(mfcc_features)
    
    logging.info("\033[1;32mBeginning training of GMM model...\033[0m")
    gmm = train_gmm(mfcc_features, n_components=args.n_components)
    
    save_gmm_model(gmm, "UBM/ubm.pkl")
    
if __name__ == "__main__":
    main()
