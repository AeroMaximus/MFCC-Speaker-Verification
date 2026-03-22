# MFCC-Speaker-Verification

This repository utalizes Gaussian mixture models (GMM) of mel frequency cepstral coefficients (MFCCs) to perform speaker verification.

By fitting a GMM to a large quantity of MFCC data of general human speech, called a universal background model (UBM), we can then adapt this model to individual speakers using a process called Maximum a Postoiri (MAP) adaptation. What this allows us to do is compare incoming data two the UBM and the adapted GMM and calculate log-likelihood scores. These scores correlate to the relative probabilities that the new data matches the distribution of data in the respective model. By comparing scores, we can determine which model most likely represents the new data, i.e. who the speech most closely resembles.

I have provided a UBM trained on a random subset of the LibriSpeech train-clean-100 dataset and an example MAP adapted GMM for the Canadian accented example.

## Tutorial
To use this repository for validating speaker similiarity, follow these steps:
1. Create environment:
    conda env create -f environment.yml
    conda activate MFCC-Speaker-Verification
2. Create a UBM with the train script if you don't want to use the one provided:
    python -m MFCC.train --audio_dir /path/to/audio/directory
3. Adapt the GMM for a specific speaker:
    python -m MFCC.adapt /path/to/UBM.pkl --audio_path /path/to/speaker/audio --save_path /path/to/save/directory
4. Run the classifiction on an audio file:
    python -m MFCC.classify /audio/file/path --gmm_0_path /path/to/UBM.pkl --gmm_1_path /path/to/speaker.pkl -return_single

