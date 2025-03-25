# Federated Learning for Small Language Models on iOS  

This repository implements a **Federated Learning (FL) system** for **Small Language Models (SLMs)**, enabling **on-device training and inference** on iOS devices. The system trains models across multiple devices while preserving user privacy by **keeping data local** and only sharing model updates.  

---

## **📌 Features**
✅ **Federated Learning Setup**: Uses TensorFlow Federated (TFF) to train SLMs across multiple clients.  
✅ **On-Device Training & Inference**: Deploys models to iOS using TensorFlow Lite (TFLite).  
✅ **FL Server for Model Aggregation**: A simple Flask-based server aggregates model updates.  

---

## **📂 Project Structure**
Federated-SLM-on-iOS/ 
├── .github/ 
│ ├── workflows/ 
│ │ ├── flutter_ios_build.yml
├── federated_slm_app/ 
│ ├── assets/ 
│ │ ├── model.tflite
│ │ ├── tokenizer.json
│ ├── ios/
│ │ ├── Podfile # CocoaPods dependencies for TensorFlow Lite 
│ ├── lib/ 
│ │ ├── tflite_mode.dart # Loads & runs TFLite model on iOS 
│ │ ├── main.dart # UI for input & model results 
│ ├── pubspec.yaml
├── server/
│ ├── app.py # FL server handling model updates 
│ ├── requirements.txt # Server dependencies 
│ ├── aggregate_model.py # Aggregates model weights from clients 
├── models/ 
│ ├── preprocess_data.py # Preprocesses text data for FL 
│ ├── load_federated_data.py # Converts preprocessed data into TFF format 
│ ├── train_federated_model.py # Federated learning training script 
│ ├── model.tflite # Trained TFLite model for iOS 



---

## Prerequisites

### 1. Install Flutter

Follow the instructions from the [Flutter installation guide](https://flutter.dev/docs/get-started/install) to install Flutter on your machine. Ensure you have a working Flutter environment set up.

- Install [Flutter SDK](https://flutter.dev/docs/get-started/install/windows).
- Set up an emulator or connect a physical device for testing.
- Install [Android Studio](https://developer.android.com/studio) (for Android) or set up Xcode (for iOS development).

### 2. Install Python

Ensure that you have Python 3.x installed on your machine. You can download and install Python from [python.org](https://www.python.org/downloads/).

- Make sure that Python is added to your system's PATH during installation.

### 3. Install Virtual Environment

It is recommended to set up a virtual environment for Python dependencies to avoid conflicts with global Python packages. Follow the steps below to set it up:

1. **Open Command Prompt**

2. **Navigate** to where you cloned this repository 

3. **Create a virtual environment** by running:

   ```bash
   python -m venv tff_new_env

4. **Activate the virtual environment:**

   ```bash
   tff_new_env\Scripts\activate

5. **Install required Python dependencies:**

   ```bash
   pip install -r requirements.txt

### 4. Install TensorFlow Lite Model

In the `models/` directory, you need to generate the TensorFlow Lite model (`model.tflite`) and tokenizer data (`tokenizer.json`). Follow these steps:

1. **Navigate to the** `models/` **folder**:

   ```bash
   cd models

2. **Generate the model and tokenizer files** by running the following Python scripts in sequence:

   ```bash
   python preprocess_data.py
   python load_federated_data.py
   python train_federated_model.py

These scripts will preprocess data, load federated data, and train a federated model, resulting in the `model.tflite` and `tokenizer.json` files.

### 5. Setting Up the Flutter App

1. **Navigate to the Flutter app directory:**

   ```bash
   cd federated_slm_app

2. **Add** `model.tflite` **and** `tokenizer.json` to the `assets/` folder of the Flutter app if they are not there already.

3. **Update** `pubspec.yaml`:

   Ensure you have the following configurations in pubspec.yaml to include the model and tokenizer in your app's assets:

   ```yaml
   flutter:
      assets:
         - assets/model.tflite
         - assets/tokenizer.json


4. **Install required Python dependencies**

   ```bash
   pip install -r requirements.txt

### 6. Install Virtual Environment
### 7. Install Virtual Environment
### 8. Install Virtual Environment
