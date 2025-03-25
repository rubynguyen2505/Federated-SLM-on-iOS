# Federated Learning for Small Language Models on iOS  

This repository implements a **Federated Learning (FL) system** for **Small Language Models (SLMs)**, enabling **on-device training and inference** on iOS devices. The system trains models across multiple devices while preserving user privacy by **keeping data local** and only sharing model updates.  

---

## **📌 Features**
✅ **Federated Learning Setup**: Uses TensorFlow Federated (TFF) to train SLMs across multiple clients.  
✅ **On-Device Training & Inference**: Deploys models to iOS using TensorFlow Lite (TFLite).  
✅ **FL Server for Model Aggregation**: A simple Flask-based server aggregates model updates.  

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Install Virtual Environment](#install-virtual-environment)
4. [Install Tensorflow Lite Model](#install-tensorflow-lite-model)
5. [Setting Up the Flutter App](#setting-up-the-flutter-app)
6. [Setting Up GitHub Actions](#setting-up-github-actions)
7. [Converting Runner.app to Runner.ipa](#converting-runner-app-to-runner-ipa)
8. [Running the Flask Server](#running-the-flask-server)
9. [Running Federated Learning](#running-federated-learning)

---

## Project Structure

```
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
```


---

## Prerequisites

### 1. Install WSL Ubuntu

You can download Ubuntu from the Microsoft Store. Once WSL Ubuntu is set up. Clone this repository to a new project directory on your WSL Ubuntu environment. 

### 2. Install Flutter

Follow the instructions from the [Flutter installation guide](https://flutter.dev/docs/get-started/install) to install Flutter on WSL Ubuntu. Ensure you have a working Flutter environment set up.

### 3. Install Python

Ensure that you have Python 3.x installed on WSL Ubuntu. You can download and install Python from [python.org](https://www.python.org/downloads/).

- Make sure that Python is added to your system's PATH during installation.

## Install Virtual Environment

It is recommended to set up a virtual environment for Python dependencies to avoid conflicts with global Python packages. Follow the steps below to set it up:

1. **Open WSL Ubuntu**

2. **Navigate** to where you cloned this repository 

3. **Create a virtual environment** by running:

   ```bash
   python3 -m venv tff_new_env

4. **Activate the virtual environment:**

   ```bash
   source tff_new_env/bin/activat

5. **Install required Python dependencies:**

   ```bash
   pip install -r requirements.txt

## Install TensorFlow Lite Model

In the `models/` directory, you need to generate the TensorFlow Lite model (`model.tflite`) and tokenizer data (`tokenizer.json`). Follow these steps:

1. **Navigate to the** `models/` **folder**:

   ```bash
   cd models

2. **Generate the model and tokenizer files** by running the following Python scripts in sequence:

   ```bash
   python3 preprocess_data.py
   python3 load_federated_data.py
   python3 train_federated_model.py

These scripts will preprocess data, load federated data, and train a federated model, resulting in the `model.tflite` and `tokenizer.json` files.

## Setting Up the Flutter App

1. **Navigate to the Flutter app directory:**

   ```bash
   cd federated_slm_app

2. **Add** `model.tflite` **and** `tokenizer.json` to the `assets/` folder of the Flutter app if they are not there already.

3. **Configure** `pubspec.yaml`:

   Ensure you have the following configurations in `pubspec.yaml`:

   ```yaml
   name: federated_slm_app
   description: A new Flutter project.
   # The following line prevents the package from being accidentally published to
   # pub.dev using `flutter pub publish`. This is preferred for private packages.
   publish_to: 'none' # Remove this line if you wish to publish to pub.dev

   # The following defines the version and build number for your application.
   # A version number is three numbers separated by dots, like 1.2.43
   # followed by an optional build number separated by a +.
   # Both the version and the builder number may be overridden in flutter
   # build by specifying --build-name and --build-number, respectively.
   # In Android, build-name is used as versionName while build-number used as versionCode.
   # Read more about Android versioning at https://developer.android.com/studio/publish/versioning
   # In iOS, build-name is used as CFBundleShortVersionString while build-number is used as CFBundleVersion.
   # Read more about iOS versioning at
   # https://developer.apple.com/library/archive/documentation/General/Reference/InfoPlistKeyReference/Articles/CoreFoundationKeys.html
   # In Windows, build-name is used as the major, minor, and patch parts
   # of the product and file versions while build-number is used as the build suffix.
   version: 1.0.0+1

   environment:
     sdk: '>=3.1.3 <4.0.0'

   # Dependencies specify other packages that your package needs in order to work.
   # To automatically upgrade your package dependencies to the latest versions
   # consider running `flutter pub upgrade --major-versions`. Alternatively,
   # dependencies can be manually updated by changing the version numbers below to
   # the latest version available on pub.dev. To see which dependencies have newer
   # versions available, run `flutter pub outdated`.
   dependencies:
     flutter:
       sdk: flutter
     tflite_flutter: ^0.10.4  # TensorFlow Lite Flutter plugin
     http: ^0.13.3


     # The following adds the Cupertino Icons font to your application.
     # Use with the CupertinoIcons class for iOS style icons.
     cupertino_icons: ^1.0.2

   dev_dependencies:
     flutter_test:
       sdk: flutter

     # The "flutter_lints" package below contains a set of recommended lints to
     # encourage good coding practices. The lint set provided by the package is
     # activated in the `analysis_options.yaml` file located at the root of your
     # package. See that file for information about deactivating specific lint
     # rules and activating additional ones.
     flutter_lints: ^2.0.0

   # For information on the generic Dart part of this file, see the
   # following page: https://dart.dev/tools/pub/pubspec

   # The following section is specific to Flutter packages.
   flutter:

     # The following line ensures that the Material Icons font is
     # included with your application, so that you can use the icons in
     # the material Icons class.
     uses-material-design: true

     # To add assets to your application, add an assets section, like this:
     # assets:
     #   - images/a_dot_burr.jpeg
     #   - images/a_dot_ham.jpeg
     assets:
       - assets/model.tflite
       - assets/tokenizer.json

     # An image asset can refer to one or more resolution-specific "variants", see
     # https://flutter.dev/assets-and-images/#resolution-aware

     # For details regarding adding assets from package dependencies, see
     # https://flutter.dev/assets-and-images/#from-packages

     # To add custom fonts to your application, add a fonts section here,
     # in this "flutter" section. Each entry in this list should have a
     # "family" key with the font family name, and a "fonts" key with a
     # list giving the asset and other descriptors for the font. For
     # example:
     # fonts:
     #   - family: Schyler
     #     fonts:
     #       - asset: fonts/Schyler-Regular.ttf
     #       - asset: fonts/Schyler-Italic.ttf
     #         style: italic
     #   - family: Trajan Pro
     #     fonts:
     #       - asset: fonts/TrajanPro.ttf
     #       - asset: fonts/TrajanPro_Bold.ttf
     #         weight: 700
     #
     # For details regarding fonts from package dependencies,
     # see https://flutter.dev/custom-fonts/#from-packages

4. **Configure** `Podfile` in `ios/` **folder**:

   Ensure you have the following configurations in `Podfile`:

   ```rb
   # Uncomment the next line to define a global platform for your project
   platform :ios, '12.0'

   # Ensure Flutter environment is loaded before using Flutter-specific pod installation
   flutter_root = File.expand_path('..', File.dirname(__FILE__))
   load File.join(flutter_root, 'ios', 'Flutter', 'podhelper.rb')

   target 'Runner' do
     # Enable dynamic frameworks to properly link TensorFlow Lite
     use_frameworks! :linkage => :dynamic

     # Pods for Runner
     pod 'TensorFlowLiteSwift'
     pod 'TensorFlowLiteC'

     # Ensure Flutter dependencies are installed
     flutter_install_all_ios_pods(File.dirname(File.realpath(__FILE__)))

     target 'RunnerTests' do
       inherit! :search_paths
       # Pods for testing
     end
   end

5. **Configure** `exportOptions.plist` in `ios/` **folder**:

   Ensure you have the following configurations in `exportOptions.plist`:
   ```bash
   <?xml version="1.0" encoding="UTF-8"?>
   <plist version="1.0">
     <dict>
       <key>method</key>
       <string>development</string>
       <key>signingStyle</key>
       <string>manual</string>
       <key>teamID</key>
       <string></string>
       <key>bundleID</key>
       <string>com.example.yourapp</string>
       <key>provisioningProfileSpecifier</key>
       <string></string>
       <key>uploadSymbols</key>
       <true/>
       <key>compileBitcode</key>
       <false/>
       <key>destination</key>
       <string>export</string>
     </dict>
   </plist>

6. Get **WSL's primary IP**:

   The first IP will be your **WSL's local IP address**

   ```bash
   hostname -I

7. Now, on **Windows (not WSL)**, run this in **Powershell**:

   ```powershell
   ipconfig
   
8. Look for the **Wireless LAN adapter Wi-Fi** or **Ethernet** section. You’ll see something like:

   ```nginx
   IPv4 Address: 192.168.12.118

9. **Forward Windows Port 5000 to WSL**

   Run this **on Windows (PowerShell as Admin)** and replace YOUR_WIFI_IP with the Windows local IP found using `ipconfig` and YOUR_WSL_IP with the WSL's local IP using `hostname -I`:

   ```powershell
   netsh interface portproxy add v4tov4 listenaddress=YOUR_WIFI_IP listenport=5000 connectaddress=YOUR_WSL_IP connectport=5000

10. **Verify**:

   ```powershell
   netsh interface portproxy show all

11. **Allow Firewall Access on Windows**:

   ```powershell
   New-NetFirewallRule -DisplayName "Allow Flask 5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow

12. Now back to the **Flutter App directory**, **Navigate** to `lib/main.dart` and **modify the following by replace** `192.168.12.118` with your **Windows local IP**:

   ```bash
   var url = Uri.parse("http://192.168.12.118:5000/get_aggregated_model");

## Setting Up GitHub Actions

1. **Navigate back to the project directory.**

2. **Navigate** to `.github/workflows/` **folder:**

   ```bash
   cd .github/workflows

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

4. **Sideload the IPA:**

   Use [Sideloadly](https://sideloadly.io/) to sideload the `Runner.ipa` onto your iPhone.

   Make sure your iPhone is running iOS 17.5 or later and Developer Mode is enabled by:
   `Settings -> Privacy & Security -> Devloper Mode -> On`

## Running the Flask Server

1. Going back to your project directory, **navigate to the** `server/` folder:

   ```bash
   cd server

2. **Run the Flask server:**

   ```bash
   python3 app.py

   The server will be hosted locally at http://127.0.0.1:5000/.

## Running Federated Learning

1. **Open the Flutter app** on your iPhone (using Sideloadly).

2. **Start the Federated Learning Simulation** by clicking the corresponding button in the app. The app will:

   a. Perform a local model update.
   
   b. Send model weights to the Flask server.

   c. Receive the aggregated model back from the serve
