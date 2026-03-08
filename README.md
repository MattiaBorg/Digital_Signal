# Automatic Drum Transcription (ADT) Pipeline

This repository contains a comprehensive pipeline for **Automatic Drum Transcription (ADT)**. The project leverages deep learning techniques to classify drum components (specifically Kick and Snare) from polyphonic audio signals. The workflow covers the entire lifecycle from data preprocessing and augmentation to model training and hyperparameter optimization.

## Project Structure
The pipeline is structured into 4 key steps:

1. **Environment Setup**: Initializes the workspace, installs essential libraries (Demucs for audio separation, Keras Tuner for optimization), and configures the directory structure to interface with Google Drive.
2. **Data Engineering & Augmentation**: Processes multi-source datasets (IDMT-SMT and Kaggle). It converts raw audio into Mel Spectrograms and applies advanced augmentation (Time Stretching, Pitch Shifting, and SpecAugment) to ensure model robustness.
3. **CNN Architecture & Hyperparameter Optimization**: Defines a dynamic Convolutional Neural Network (CNN) architecture. It uses Keras Tuner (Hyperband) to automate the search for optimal hyperparameters (filters, dropout, learning rate) and implements callbacks like EarlyStopping and ReduceLROnPlateau.
4. **Audio Signal Analysis**: Performs exploratory analysis of target audio tracks using time-domain (waveform) and frequency-domain (log-spectrogram) visualizations to assess signal characteristics before transcription.

## Requirements
The project is built for **Google Colab**. Key dependencies include:

```bash
pip install demucs keras-tuner scikit-learn pretty_midi librosa tensorflow matplotlib seaborn tqdm
