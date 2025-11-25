AI Therapist — Emotion Recognition Model
Overview

This project implements an AI-based therapist that analyzes user emotions using facial-expression recognition.
The system uses a deep-learning model trained on the FER-2013 dataset to classify emotions and generate supportive, 
context-aware responses. The repository contains the full source code, preprocessing pipeline, model architecture,
and inference scripts.

Note:
The dataset and trained model files are not included in this repository due to their large size. Follow the steps 
below to set up the project correctly.

Dataset Requirement (Important)

This project requires the FER-2013 dataset for training and testing.
Anyone who wants to run the code must download and place the dataset manually.

Download the FER-2013 dataset from Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013

Extract the dataset.

Place the extracted folder into the project directory following this structure:

/data/FER-2013/


Ensure that the CSV/Images match the same folder names expected by the preprocessing code.
Once added, the training and inference scripts will run without modifications.

Features

1) Emotion recognition using CNN-based deep-learning models
2) Support for key emotions (happy, sad, angry, neutral, etc.)
3) Real-time prediction pipeline (via webcam or image input)
4) Simple conversational logic for emotion-aware responses
5) Modular code structure for easy extension or model retraining
