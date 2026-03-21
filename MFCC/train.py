import logging
import os
from sklearn.mixture import GaussianMixture
import numpy as np
from MFCC.classification import extract_mfcc
from MFCC.common import find_files_with_extension, save_gmm_model
import datetime
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

def train_gmm(features, n_components: int = 6, reg_covar=1e-6) -> GaussianMixture:
    # Check if features are a 2D array
    if not isinstance(features, np.ndarray) or len(features.shape) != 2:
        raise ValueError("Features must be a 2D numpy array.")
    
    # Check for NaN or infinite values
    if np.isnan(features).any() or np.isinf(features).any():
        logging.error("Features contain NaNs or infinities. Please check the data preprocessing.")
        raise ValueError("Features contain invalid values.")
    
    logging.debug(f"First few samples:\n{features[:5]}")

    try:
        gmm = GaussianMixture(n_components=n_components, random_state=0,
                              warm_start=True, verbose=2, reg_covar=reg_covar,
                              covariance_type='diag')
        gmm.fit(features)

    except Exception as e:
        logging.error(f"Failed to fit GMM model: {str(e)}")
        raise

    return gmm
   
def main():
    parser = argparse.ArgumentParser(description="GMM generator")
    parser.add_argument("--audio_dir", type=str, help="Path to the audio directory to train on")
    parser.add_argument("--n_components", type=int, help="Number of Gaussian Mixture components", default=512)
    parser.add_argument("--n_mfcc", type=int, help="Number of MFCC features to extract", default=13)
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
            logging.info(f"Loaded MFCC features shape: {mfcc_features.shape}")
        except Exception as e:
            logging.error(f"Failed to load pre-extracted MFCC features from {args.pre_extracted}: {str(e)}")
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
    
    # Reduce the dataset size for debugging purposes
    if mfcc_features.shape[0] > 1_000_000:
        logging.warning("Dataset is very large. Using a subset for debugging.")
        subset_size = 1_000_000
        mfcc_features = mfcc_features[:subset_size, :]
    
    logging.info("\033[1;32mBeginning training of GMM model...\033[0m")
    gmm = train_gmm(mfcc_features, n_components=args.n_components)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_gmm_model(gmm, f"MFCC/UBM/ubm_{args.n_components}_{timestamp}.pkl")
    
if __name__ == "__main__":
    main()
