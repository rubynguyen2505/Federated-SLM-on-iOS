# Federated Learning for Small Language Models on iOS  

This repository implements a **Federated Learning (FL) system** for **Small Language Models (SLMs)**, enabling **on-device training and inference** on iOS devices. The system trains models across multiple devices while preserving user privacy by **keeping data local** and only sharing model updates.  

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Install Virtual Environment](#install-virtual-environment)
4. [Training and Converting the TensorFlow Lite Model](#training-and-converting-the-tensorflow-lite-model)
5. [Setting Up the Flutter App](#setting-up-the-flutter-app)
6. [Setting Up GitHub Actions for CI/CD](#setting-up-github-actions-for-cicd)
7. [Converting Runner.app to Runner.ipa](#converting-runnerapp-to-runneripa)
8. [Running the Flask Server](#running-the-flask-server)
9. [Running Federated Learning](#running-federated-learning)

---

## Project Structure

```
├── federated_slm_app/ 
│ ├── ios/
│ │ ├── Runner/
│ │ │ ├── imdb_updatable_model.mlpackage # The updatable base model for federated learning
│ │ │ ├── AppDelegate.swift # Client logic for handling on-device training/inference
│ │ │ ├── tokenizer.json # Distributed tokenizer for tokenizing text inputs
│ │ ├── Podfile # CocoaPods dependencies 
│ ├── lib/ 
│ │ ├── main.dart # UI for input & model results 
│ ├── pubspec.yaml # Flutter specifications and dependencies
├── server/
│ ├── models/ # Extracted models from clients
│ ├── uploads/ # Uploaded compressed models from clients 
│ ├── app.py # FL server handling distributed training/test data, obtaining metrics, and model aggregation
│ ├── aggregated_model.mlmodel # Global model aggregated from clients 
│ ├── benchmark.py # Visualize metrics 
│ ├── imdb_model.keras # Centralized model for comparison
├── requirements.txt # Server dependencies 

```


---

## Prerequisites

**Please read through this Prerequisites section before you clone this repository**

To setup the project, you will need a macOS machine and an iPhone running on iOS 16.0 or later with an active AppleID. For the macOS machine, it is recommended that you have a physical machine that runs on macOS operating system for ease of setup. If you only have a Windows machine, then we highly recommend that you use VMWare to host a virtual machine that runs on macOS, just like how we did. Please refer to these YouTube guides to setup a macOS virtual machine on your Windows physical machine:

For Intel: https://youtu.be/Fq6j9CS7C5g?si=lfUbLvTTYuZOxFlc

For AMD: https://youtu.be/gY97OI-bTxE?si=FYskvw_nN0MXH1Qt

Note that if you have a physical macOS machine, then setting up the server and building the iOS app will be done on the same codebase (you just need to clone this project once). Otherwise, if you have a macOS virtual machine, you'll need to clone this project twice. Once on your Windows environment with WSL Ubuntu to setup the server, and once on your macOS virtual machine to build the iOS app.

### 1. Install Xcode

We assume that you have your macOS machine up and running. If you have macOS as a virtual machine, then it is likely macOS Sonoma you are using if you followed the YouTube guides. We suggest that you get Xcode 15 as it is compatible with this macOS version. Otherwise, if you have a physical macOS machine, then install the Xcode version that is compatible with the macOS version you are running. Please visit this Apple's Developer website to download Xcode to your macOS machine: https://developer.apple.com/download/all/?q=Xcode

Once you have downloaded Xcode and extracted it as an application file, you will see something like this when Xcode is opened for the first time. Make sure to check the iOS platform for installation:

![Screenshot](guide_images/xcode.png)

### 2. Install Homebrew, Git, and CocoaPods

Git makes it easy to install and work with Flutter and building the iOS app later on while CocoaPods will make the process of managing dependencies on the Xcode project easier. To install Git and CocoaPods the easy way, we need to install Homebrew first on our macOS machine. 

Follow the instructions from the [Homebrew installation guide](https://brew.sh/) to install Homebrew and add it to PATH on your macOS machine. Once you have successfully installed Homebrew and added it to PATH, you should be able to run this command on your macOS terminal:

```bash
brew --version
```

Next, you will install Git and CocoaPods and add them to PATH via brew command, this can be done as followed:

```bash
brew install git
brew install cocoapods
```

Once you have successfully installed Git and CocoaPods and added them to PATH, you should be able to run these commands:

```bash
git --version
pod --version
```

### 3. Install VSCode

If you have macOS as a virtual machine, then you would need to have VSCode on both the Windows physical machine and the macOS virtual machine. Otherwise, you would only need to install VSCode once if it is a physical macOS machine.
Follow the instructions from the [VSCode installation guide for Windows](https://code.visualstudio.com/docs/setup/windows) and the [VSCode installation guide for macOS](https://code.visualstudio.com/docs/setup/mac) to install VSCode and add it to PATH. 

### 4. Install Flutter

Flutter is needed to build our iOS app. Follow the instructions from the [Flutter installation guide](https://docs.flutter.dev/get-started/install/macos/mobile-ios) to install Flutter and add it to PATH on your macOS machine. 

Once you have Flutter installed and added to PATH, you should be able to run this command on your macOS terminal:

```bash
flutter doctor
```

Make sure the summary should look something like this. All but the Android toolchain and development for the web sections should have a check mark before them. If you see the Xcode section have a cross mark, follow the instructions on the terminal to complete Xcode setup, then run 'flutter doctor' command again.

![Screenshot](guide_images/flutter_doctor.png)

### 5. Install Python

Python is needed to setup the server side of the project. If you have a physical macOS machine, then you can simply use brew to install Python on it using the following command:

```bash
brew install python
```

Once Python is installed and added to PATH on your physical macOS machine, you should be able to run this command on your macOS terminal:

```bash
python3 --version
```

If you have macOS as a virtual machine, then simply having Python installed on the Windows environment is enough. You can download and install Python on your Windows machine from [python.org](https://www.python.org/downloads/).

Make sure that Python is added to your system's PATH during installation. You can check with this command on your Wndows terminal:

```
python --version
```

## Clone the repository

If you have either a physical or virtual macOS machine, open up a Terminal and clone the repository using this command:

```bash
git clone https://github.com/rubynguyen2505/Federated-SLM-on-iOS.git
```

If you are using a macOS virtual machine, additionally run the same command on a Terminal on your Windows machine.

## Install Virtual Environment

Since our server is a Python Flask-based server, it is recommended to set up a virtual environment for Python dependencies to avoid conflicts with global Python packages. If you have a macOS virtual machine, only open up a Terminal on the Windows environment since that is where we setup the server. If you have a physical macOS machine, then open up a Terminal on it instead. Follow the steps below to set it up:

1. **Navigate** to where you cloned this repository 

2. **Create a virtual environment** (if on physical macOS):

   ```bash
   python3 -m venv tff_new_env
   ```

   or if on Windows:

   ```bash
   python -m venv tff_new_env
   ```

3. **Activate the virtual environment** (if on physical macOS):

   ```bash
   source tff_new_env/bin/activate
   ```

   or if on Windows:

   ```bash
   tff_new_env\Scripts\activate
   ```

5. **Install required Python dependencies** (if on physical macOS):

   ```bash
   pip3 install -r requirements.txt
   ```

   or if on Windows:

   ```bash
   pip install -r requirements.txt

## Running the Flask server

1. **Navigate** to the `server/` folder:

   ```bash
   cd server
   ```

2. **Run the Flask server** 

   ```bash
   python3 app.py
   ```

   The server will be hosted locally at http://127.0.0.1:5000/.

## Setting Up the Flutter App

1. **Navigate to the Flutter app directory:**

   ```bash
   cd federated_slm_app
   ```

2. **Add** `model.tflite` **and** `tokenizer.json` to the `assets/` folder of the Flutter app if they are not there already.


6. Get **WSL's primary IP**:

   The first IP will be your **WSL's local IP address**

   ```bash
   hostname -I
   ```

7. Now, on **Windows (not WSL)**, run this in **Powershell**:

   ```powershell
   ipconfig
   ```
   
8. Look for the **Wireless LAN adapter Wi-Fi** or **Ethernet** section. You’ll see something like:

   ```nginx
   IPv4 Address: 192.168.12.118
   ```

9. **Forward Windows Port 5000 to WSL**

   Run this **on Windows (PowerShell as Admin)** and replace YOUR_WIFI_IP with the Windows local IP found using `ipconfig` and YOUR_WSL_IP with the WSL's local IP using `hostname -I`:

   ```powershell
   netsh interface portproxy add v4tov4 listenaddress=YOUR_WIFI_IP listenport=5000 connectaddress=YOUR_WSL_IP connectport=5000
   ```

10. **Verify**:

   ```powershell
   netsh interface portproxy show all
   ```

11. **Allow Firewall Access on Windows**:

   ```powershell
   New-NetFirewallRule -DisplayName "Allow Flask 5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
   ```

12. Now back to the **Flutter App directory**, **Navigate** to `lib/tflite_model.dart` and **modify the following by replace** `192.168.12.118` with your **Windows local IP**:

   ```bash
   var url = Uri.parse("http://192.168.12.118:5000/get_aggregated_model");
   ```

## Setting Up GitHub Actions for CI/CD

1. **Navigate back to the project directory.**

2. **Navigate** to `.github/workflows/` **folder:**

   ```bash
   cd .github/workflows
   ```

3. **Configure** `flutter_ios_build.yml`:

   Ensure you have the following configurations in `flutter_ios_build.yml`:

   ```yml
   name: Build iOS App for iOS without signing

   on:
     push:
       branches:
         - master
     pull_request:
       branches:
         - master

   jobs:
     build:
       runs-on: macos-latest  # Use macOS runner

       steps:
         # Checkout code from the repository
         - name: Checkout code
           uses: actions/checkout@v3

         # Set up Flutter
         - name: Set up Flutter
           uses: subosito/flutter-action@v2
           with:
             flutter-version: '3.13.6'

         # Clean previous builds (optional but recommended)
         - name: Clean previous builds
           run: |
             cd federated_slm_app
             flutter clean

         # Install dependencies (flutter packages)
         - name: Install Flutter dependencies
           run: |
             cd federated_slm_app
             flutter pub get
         
         # Install iOS dependencies (CocoaPods)
         - name: Install iOS CocoaPods dependencies
           run: |
             cd federated_slm_app
             cd ios
             pod install

         # Build iOS app without signing
         - name: Build iOS app without signing
           run: |
             cd federated_slm_app
             flutter build ios --no-codesign

         # Archive the iOS app (no signing)
         - name: Archive iOS app
           run: |
             cd federated_slm_app
             cd ios
            xcodebuild -workspace Runner.xcworkspace -scheme Runner -configuration Release -archivePath $PWD/build/Runner.xcarchive archive -allowProvisioningUpdates CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO

         # List contents of the .xcarchive to confirm .app exists
         - name: List contents of .xcarchive
           run: |
             cd federated_slm_app/ios
             ls -R $PWD/build/Runner.xcarchive/Products/Applications/

         # Upload the extracted .app as artifact
         - name: Upload .app as artifact
           uses: actions/upload-artifact@v4
           with:
             name: runner-app
             path: federated_slm_app/ios/build/Runner.xcarchive/Products/Applications/Runner.app
   ```

4. **Push to GitHub:**

   Push your code to GitHub, triggering the CI/CD pipeline. The build process will generate an artifact `runner-app.zip` that can be downloaded.

5. **Download and extract the build artifact:**

   After the build completes, download the artifact (`runner-app.zip`) from GitHub Actions and extract it to an empty folder on your Windows machine. Name that folder as `Runner.app`.

## Converting Runner.app to Runner.ipa

1. **Download** `apptoipa.exe` from the following link:

   https://github.com/deqline/IPABundler

2. Place `apptoipa.exe` **in the same directory** as `Runner.app`.

3. Convert `Runner.app` to `Runner.ipa` by running the following command in the Command Prompt:

   ```bash
   apptoipa Runner.app
   ```

4. **Sideload the IPA:**

   Use [Sideloadly](https://sideloadly.io/) to sideload the `Runner.ipa` onto your iPhone.

   Make sure your iPhone is running iOS 17.5 or later and Developer Mode is enabled by:
   `Settings -> Privacy & Security -> Devloper Mode -> On`

## Running the Flask Server

1. Going back to your project directory, **navigate to the** `server/` folder:

   ```bash
   cd server
   ```

2. **Run the Flask server:**

   ```bash
   python3 app.py
   ```

   The server will be hosted locally at http://127.0.0.1:5000/.

## Running Federated Learning

1. **Open the Flutter app** on your iPhone (using Sideloadly).

2. **Start the Federated Learning Simulation** by clicking the corresponding button in the app. The app will:

   a. Perform a local model update.
   
   b. Send model weights to the Flask server.

   c. Receive the aggregated model back from the serve

3. **Run the `TEST.ipynb` file in `server/` folder on Google Colab to see benchamark results**
