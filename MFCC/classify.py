import logging
from sklearn.mixture import GaussianMixture
import numpy as np
import argparse
import matplotlib.pyplot as plt
from MFCC.common import load_gmm_model, extract_mfcc

def log_likelihood_audio_frames(mfcc_features: np.ndarray, gmm_models: list[GaussianMixture]) -> np.ndarray:
    # Check if the input is a 2D array
    if mfcc_features.ndim != 2:
        logging.error("Input mfcc_features must be a 2D array")
        raise ValueError("Input mfcc_features must be a 2D array")
    
    # Check that gmm_models is populated
    if gmm_models.count == 0:
        logging.error("No Gaussian Mixture Models were entered")
        raise ValueError("No Gaussian Mixture Models were entered")
    
    feature_shape = mfcc_features.shape[1]

    for i, gmm_model in enumerate(gmm_models):
        if feature_shape != np.asarray(gmm_model.means_).shape[1]:
            logging.error(f"Number of MFCC features does not match the GMM model for speaker {i}")
            raise ValueError(f"Number of MFCC features does not match the GMM model for speaker {i}")

    log_likelihoods = np.array([gmm_model.score_samples(mfcc_features) for gmm_model in gmm_models])

    return log_likelihoods

def classify_audio_average(log_likelihood_array: np.ndarray) -> int:
    
    # Check if the input is a 2D array
    if log_likelihood_array.ndim != 2:
        logging.error("Input log_likelihood_array must be a 2D array")
        raise ValueError("Input log_likelihood_array must be a 2D array")

    # Average each row
    average_log_likelihoods = np.mean(log_likelihood_array, axis=1)
    logging.info(f"Average log-likelihood scores calculated for each speaker: {average_log_likelihoods}")
    
    # Find the speaker with the highest average log likelihood
    most_likely_speaker:int = np.argmax(average_log_likelihoods, axis=0)
    
    return most_likely_speaker

def classify_audio_per_frame(log_likelihood_array: np.ndarray) -> list[int]:
    
    # Check if the input is a 2D array
    if log_likelihood_array.ndim != 2:
        logging.error("Input log_likelihood_array must be a 2D array")
        raise ValueError("Input log_likelihood_array must be a 2D array")
    
    most_likely = np.argmax(log_likelihood_array, axis=0) # Max value index per column
    
    return most_likely.tolist()

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Classify audio between two trained Gaussian Mixture Models")
    parser.add_argument("audio_path", help = "Path to the input audio file")
    parser.add_argument("--gmm_0_path", required = True, help = "Path to the GMM model for speaker 0")
    parser.add_argument("--gmm_1_path", required = True, help = "Path to the GMM model for speaker 1")
    parser.add_argument("-return_single", help= "Return only a single result for the entire audio sequence", action="store_true")
    parser.add_argument("-v", help="Enable verbose logging", action="store_true")

    args = parser.parse_args()

    # Adjust logging level based on the --verbose flag
    log_level = logging.INFO if args.v else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Import GMMs for speakers 0 and 1 from provided arguments
    gmm0 = load_gmm_model(args.gmm_0_path)
    gmm1 = load_gmm_model(args.gmm_1_path)
    logging.info("GMMs loaded")

    # Extract MFCC features from the provided audio file
    mfcc_features = extract_mfcc(args.audio_path)
    logging.info("MFCCs extracted")

    log_likelihoods = log_likelihood_audio_frames(mfcc_features, [gmm0,gmm1])
    logging.info("Log-likelihoods calculated")
    
    if not args.return_single:
    
        # Initialize speaker verification output
        output = np.zeros(len(mfcc_features), dtype=int)

        output = classify_audio_per_frame(log_likelihoods)

        # Plot the results
        plt.figure(figsize=(12, 4))
        plt.plot(output, drawstyle='steps-post')
        plt.title('Speaker Identification Over Time')
        plt.xlabel('Frame Index (Time)')
        plt.ylabel('Speaker Detected')
        plt.grid(True)
        plt.show()
        
    else:
        print(classify_audio_average(log_likelihoods))

if __name__ == "__main__":
    main()