# MFCC-Speaker-Verification

This repository utilizes Gaussian Mixture Models (GMMs) of Mel Frequency Cepstral Coefficients (MFCCs) to perform speaker verification.

## Overview

By fitting a GMM to a large quantity of MFCC data of general human speech, called a Universal Background Model (UBM), we can adapt this model to individual speakers using Maximum A Posteriori (MAP) adaptation. This process allows us to compare incoming audio data against the UBM and adapted GMMs by calculating log-likelihood scores. These scores correlate with the probabilities that new data matches each respective model, enabling us to determine which model most closely resembles the incoming speech.

I've provided a pre-trained UBM based on a random subset of the LibriSpeech train-clean-100 dataset.

## Tutorial

To use this repository for speaker verification, follow these steps:

1. **Create Environment**
   ```bash
   conda env create -f environment.yml
   conda activate MFCC-Speaker-Verification
   ```

2. **Create a UBM** (Optional: if you don't want to use the provided one)
   ```bash
   python -m MFCC.train --audio_dir /path/to/audio/directory
   ```

3. **Adapt the GMM for a Specific Speaker**
   ```bash
   python -m MFCC.adapt /path/to/UBM.pkl --audio_path /path/to/speaker/audio --save_path /path/to/save/directory
   ```

4. **Run Classification on an Audio File**
   ```bash
   python -m MFCC.classify /audio/file/path --gmm_0_path /path/to/UBM.pkl --gmm_1_path /path/to/speaker.pkl
   ```
