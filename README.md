# Sign Language Detection using Machine Learning

This project implements a sign language detection system using OpenCV, Mediapipe, and a RandomForestClassifier trained on hand landmarks. The system captures hand gestures, extracts features, and classifies them into predefined categories.

## Table of Contents
- [Installation](#installation)
- [Dataset Collection](#dataset-collection)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Warning](#warning)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kabirkohli123/Sign-language-detection-with-Python-and-Scikit-Learn
   cd Sign-language-detection-with-Python-and-Scikit-Learn

   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Collection
To collect images for training:
1. Run `collect_imgs.py` to capture hand gesture images:
   ```bash
   python collect_imgs.py
   ```
2. Follow on-screen instructions to collect images for different classes.

## Training the Model
1. Run `create_dataset.py` to process images and extract hand landmarks:
   ```bash
   python create_dataset.py
   ```
2. Train the classifier using `train_classifier.py`:
   ```bash
   python train_classifier.py
   ```
   The trained model will be saved as `model.p`.

## Running Inference
To test the sign language detection:
```bash
python inference_classifier.py
```
Press `q` to exit the video feed.

## Project Structure
```
SignLanguageDetection/
│-- collect_imgs.py        # Collects images for dataset
│-- create_dataset.py      # Extracts hand landmarks
│-- train_classifier.py    # Trains the machine learning model
│-- inference_classifier.py# Runs real-time sign detection
│-- model.p                # Trained model file
│-- data/                  # Folder for collected images
│-- requirements.txt       # Dependencies
```

## Requirements
- Python 3.10+
- OpenCV
- Mediapipe
- Scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Warning
⚠ **When testing the project, only one hand should be in front of the camera.** If a second hand appears in the frame, the program may throw an error and not work correctly.

