# A Comprehensive Approach to Deepfake Detection Using Image Processing and AI

## Overview

Deepfakes, AI-generated synthetic media, pose significant challenges to digital authenticity. This project implements and evaluates a deep learning model designed to detect deepfake videos effectively. The approach utilizes a cascade architecture combining a ResNeXt-50 CNN for spatial feature extraction and a Bidirectional LSTM (BiLSTM) for temporal sequence analysis to identify subtle inconsistencies indicative of manipulation. The model is trained and evaluated on standard benchmark datasets like FaceForensics++ and Celeb-DF.

## Key Features

* Deepfake video detection using a ResNeXt-50 + BiLSTM architecture.
* Code for training on the FaceForensics++ dataset.
* Code for fine-tuning the model on the Celeb-DF dataset for improved generalization.
* Evaluation scripts for assessing model performance.
* A user-friendly interface using Streamlit for easy video upload and prediction.

## Performance Highlights

* Achieved **91.43%** accuracy on the FaceForensics++ test set.
* Achieved **96.26%** accuracy on the Celeb-DF test set after fine-tuning (with 99.81% accuracy on fake videos).
* The model architecture prioritizes a balance between detection performance and computational efficiency.
