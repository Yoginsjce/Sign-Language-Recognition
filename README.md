# Sign-Language-Recognition
# ✋ Sign Language Recognition System

A real-time hand gesture recognition system built using **OpenCV**, **MediaPipe**, and **scikit-learn**, designed to identify and classify sign language gestures through webcam input. This project demonstrates the integration of computer vision and machine learning for an assistive technology application.

## 🚀 Project Overview

This project focuses on recognizing **static hand gestures** from sign language using computer vision techniques and an SVM-based machine learning model. It captures hand landmarks using MediaPipe, extracts feature vectors, trains an SVM classifier, and provides real-time predictions through webcam integration.

## ✨ Recognized Gestures

The system currently supports recognition of the following hand gestures:
- **HOUSE**   
- **I LOVE YOU**  
- **YES**
- **NO**  
- **HELLO**
- **LIKE**
- **VICTORY**
- **OK**
- **SUPER**
- **NICE TO MEET YOU**
- **THANKS**

You can expand this list by collecting more images and retraining the model.

## 📊 Model Accuracy

- **Classifier**: Support Vector Machine (SVM) with RBF kernel  
- **Training Accuracy**: ~**98.5%**  
- **Testing Accuracy**: ~**96.7%**  
- **Evaluation Metrics**: Precision, Recall, F1-Score, Confusion Matrix (visualized)

> The model was evaluated on a stratified train-test split (70-30) using scikit-learn.

## 🖼️ Dataset Details

- Each gesture class has **100+ labeled images**, captured using a webcam.  
- Total dataset size: **~1000+ images** across 11 gesture classes.  
- Each image is converted to a **63-dimensional vector** based on hand landmarks using **MediaPipe**.

## 🔧 Features

- 🎥 Real-time webcam-based gesture detection and classification  
- 🖐️ Hand landmark extraction using MediaPipe (21 points × 3 coordinates = 63 features)  
- 📸 Automated dataset creation using OpenCV  
- 📊 Model evaluation using precision, recall, F1-score, and confusion matrix  
- 🧠 SVM classifier optimized for multi-class classification  
- 💾 Pickle-based model saving and loading  
- 📂 CSV dataset generation for reproducible training

## 🛠️ Tech Stack

- **Programming Language**: Python  
- **Libraries Used**:
  - `OpenCV` – Image capture & processing  
  - `MediaPipe` – Hand landmark detection  
  - `scikit-learn` – Model training & evaluation  
  - `NumPy`, `Pandas` – Data handling  
  - `Matplotlib`, `Seaborn` – Visualization


