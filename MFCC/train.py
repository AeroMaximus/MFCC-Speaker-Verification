import logging
from sklearn.mixture import GaussianMixture
import numpy as np
from classification import extract_mfcc
from common import find_files_with_extension, npy_to_hash

def train_gmm(features, n_components: int = 6, reg_covar=1e-6) -> GaussianMixture:
    # Check if features are a 2D array
    if not isinstance(features, np.ndarray) or len(features.shape) != 2:
        raise ValueError("Features must be a 2D numpy array.")
    
    logging.debug(f"Shape of features: {features.shape}")
    # logging.debug(f"First few samples:\n{features[:5]}")
    
    gmm = GaussianMixture(n_components=n_components, random_state=0, reg_covar=reg_covar, covariance_type='diag')
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
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
   
def main():
    parser = argparse.ArgumentParser(description="GMM generator")
    parser.add_argument("--audio_dir", type=str, help="Path to the audio directory to train on")
    parser.add_argument("--n_components", type=int, help="Number of Gaussian Mixture components", default=512)
    parser.add_argument("--n_mfcc", type=int, help="Number of MFCC features to extract", default=39)
    parser.add_argument("--pre-extracted", type=str, help="Path to pre-extracted MFCC features .npy file")
    parser.add_argument("-v", action='store_true', help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.audio_dir is None and args.pre_extracted is None:
        parser.error("At least one of --audio_dir or --pre-extracted is required.")
        
    # Adjust logging level based on the --verbose flag
    log_level = logging.DEBUG if args.v else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        
    logging.info("\033[1;32mFinding Audio Files...\033[0m")
    
    mfcc_features = None
    npy_path = None
    
    if args.pre_extracted:
        try:
            npy_path = args.pre_extracted
            mfcc_features = np.load(npy_path)
        except:
            logging.error(f"Failed to load pre-extracted MFCC features from {args.pre_extracted}.")
            raise
    else:
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
        
        # Save the MFCC features to a file in case GMM training fails
        try:
            npy_path = args.audio_dir + "/mfcc_features.npy"
            np.save(npy_path, mfcc_features)
        except Exception as e:
            logging.warning(f"Failed to save extracted features: {str(e)}")
    
    logging.info("\033[1;32mBeginning training of GMM model...\033[0m")
    gmm = train_gmm(mfcc_features, n_components=args.n_components)
    
    hash = npy_to_hash(npy_path)
    save_gmm_model(gmm, f"UBM/ubm_{hash}.pkl")
    
if __name__ == "__main__":
    main()
