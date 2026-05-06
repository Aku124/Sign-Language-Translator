# 🎙️ Task-Aware Real-Time Sign Language Translator

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen)

## Overview
This project is an end-to-end, full-stack machine learning application that translates physical sign language gestures into spoken text in real-time. 

To overcome the latency and environmental overfitting issues common in heavy Convolutional Neural Networks (CNNs), this system completely decouples the video feed from the AI. It utilizes **Google MediaPipe** to extract 21 precise 3D spatial coordinates from the user's hand. These mathematical distances are then passed to an optimized **Random Forest Classifier**, achieving 99% accuracy on consumer hardware with zero lag.

The entire pipeline is deployed via a local **Flask** web server, featuring an anti-flicker stabilization algorithm and thread-safe Text-to-Speech (TTS) integration.

## ✨ Key Features
* **Geometric Feature Extraction:** Uses MediaPipe to track hand joints rather than processing heavy raw image pixels, ensuring real-time performance.
* **Custom Burst-Mode Dataset:** Includes a custom Python script to rapidly generate hyper-personalized coordinate datasets, eliminating background/lighting bias.
* **Anti-Flicker Stabilization:** Custom debounce logic enforces a 10-frame stability threshold before accepting a gesture, preventing screen jitter.
* **Live Web Dashboard:** A responsive, dark-mode web UI built with HTML/CSS and served via Flask MJPEG streaming.
* **Thread-Safe Audio:** Integrates `pyttsx3` for offline text-to-speech, engineered to prevent driver crashes during asynchronous web requests.

## 🛠️ Technology Stack
* **Backend Framework:** Flask
* **Computer Vision:** OpenCV (`cv2`), Google MediaPipe
* **Machine Learning:** Scikit-Learn (Random Forest), Pandas
* **Audio Engine:** Pyttsx3 (Native OS offline TTS)
* **Frontend:** HTML, CSS, JavaScript (Fetch API)

## 🚀 How to Run the Web Application

**1. Install Dependencies**
Ensure you have Python installed, then run:
```bash
pip install flask opencv-python mediapipe pandas scikit-learn pyttsx3
