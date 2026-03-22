import logging
import numpy as np
from collections import deque
import argparse
import matplotlib.pyplot as plt
from MFCC.common import load_gmm_model, extract_mfcc
from MFCC.classify import classify_audio_average, log_likelihood_audio_frames

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

    # Initialize speaker verification output
    output = np.zeros(len(mfcc_features), dtype=int)

    # Initialize audio buffer
    buffer_size: int = 2
    audio_buffer = FrameBuffer(max_size = buffer_size)

    # Loop through the audio
    for (i, frame) in enumerate(mfcc_features):
        audio_buffer.add_frame(frame)
        current_buffer = audio_buffer.get_buffer()
        log_likelihoods = log_likelihood_audio_frames(current_buffer, [gmm0,gmm1])
        output[i] = classify_audio_average(log_likelihoods)

    # Plot the results
    plt.figure(figsize=(12, 4))
    plt.plot(output, drawstyle='steps-post')
    plt.title('Speaker Identification Over Time')
    plt.xlabel('Frame Index (Time)')
    plt.ylabel('Speaker Detected')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()