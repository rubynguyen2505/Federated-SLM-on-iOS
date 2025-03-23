# Federated Learning for Small Language Models on iOS  

This repository implements a **Federated Learning (FL) system** for **Small Language Models (SLMs)**, enabling **on-device training and inference** on iOS devices. The system trains models across multiple devices while preserving user privacy by **keeping data local** and only sharing model updates.  

---

## **📌 Features**
✅ **Federated Learning Setup**: Uses TensorFlow Federated (TFF) to train SLMs across multiple clients.  
✅ **On-Device Training & Inference**: Deploys models to iOS using TensorFlow Lite (TFLite).  
✅ **FL Server for Model Aggregation**: A simple Flask-based server aggregates model updates.  
✅ **Text Processing for SLMs**: Prepares datasets like Shakespeare or mobile text input for training.  

---

## **📂 Project Structure**
Federated-SLM-on-iOS/ ├── server/
│ ├── app.py # FL server handling model updates │ ├── requirements.txt # Server dependencies │ ├── aggregate_model.py # Aggregates model weights from clients ├── models/ │ ├── preprocess_data.py # Preprocesses text data for FL │ ├── load_federated_data.py # Converts preprocessed data into TFF format │ ├── train_federated_model.py # Federated learning training script │ ├── model.tflite # Trained TFLite model for iOS ├── iOSApp/ │ ├── Podfile # CocoaPods dependencies for TensorFlow Lite │ ├── iOSApp.xcodeproj/ # Xcode project │ ├── Model/ │ │ ├── TFLiteModel.swift # Loads & runs TFLite model on iOS │ ├── Views/ │ │ ├── ContentView.swift # UI for input & model results │ ├── FederatedUpdates/ │ │ ├── LocalTraining.swift # Runs on-device training │ │ ├── ServerCommunication.swift # Communicates with FL server


---

## **🚀 Getting Started**
### **🔧 Requirements**
- **Python 3.8+**
- **TensorFlow & TensorFlow Federated (TFF)**
- **Flask (for the FL server)**
- **Xcode (for iOS deployment)**
- **TensorFlow Lite for Swift (installed via CocoaPods)**

### **1️⃣ Setting Up the FL Server**
1. Install dependencies:  
   ```sh
   cd server
   pip install -r requirements.txt
