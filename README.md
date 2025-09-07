# A Comprehensive Approach to Deepfake Detection Using Image Processing and AI

## Overview

Deepfakes, AI-generated synthetic media, pose significant threats to digital authenticity, trust, and security. Detecting such manipulations is essential for safeguarding information integrity in journalism, social media, and forensic investigations.

This project implements and evaluates a deep learning–based deepfake detection system capable of identifying subtle spatial and temporal artifacts present in manipulated videos. The approach utilizes a cascade architecture:
* ResNeXt-50 CNN for high-level spatial feature extraction from individual frames.
* Bidirectional LSTM (BiLSTM) for temporal sequence modeling across consecutive frames, enabling the detection of frame-level inconsistencies often overlooked by single-frame methods.
The model is trained and tested on benchmark datasets such as FaceForensics++ and Celeb-DF, ensuring both robustness and generalization across different deepfake generation techniques.

## Key Features

* End-to-end deepfake video detection using a ResNeXt-50 + BiLSTM hybrid architecture.
* Comprehensive data preprocessing pipeline including frame extraction, face detection, resizing, and normalization.
* Code for training on the FaceForensics++ dataset, with provisions for reproducible experiments.
* Code for fine-tuning the model on the Celeb-DF dataset to improve performance on diverse real-world manipulations.
* Evaluation scripts to compute metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC.
* A Streamlit-based interactive interface for uploading videos and obtaining real-time predictions.
* Modular code structure for easy extension with other CNN or transformer backbones.

## Performance Highlights

* Achieved 91.43% accuracy on the FaceForensics++ test set.
* Achieved 96.26% accuracy on the Celeb-DF test set after fine-tuning.
* Achieved 99.81% detection accuracy on fake videos in the Celeb-DF dataset, demonstrating high sensitivity to manipulations.
* The architecture ensures a balance between detection performance and computational efficiency, making it suitable for research as well as potential real-time applications.

## Future Work

* Integration of transformer-based architectures (e.g., Vision Transformers, TimeSformer) for improved temporal modeling.
* Development of a lightweight mobile-friendly model for on-device deepfake detection.
* Exploration of multimodal detection by incorporating both audio and video signals.
* Deployment of a web-based demo with cloud-backed scalability.
