import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:federated_slm_app/tflite_model.dart';
import 'package:http/http.dart' as http;  // Import the HTTP package

// Function to send model weights to the server
Future<bool> sendModelWeights(List<double> weights) async {
  var url = Uri.parse("http://172.24.201.114:5000/upload_model");

  try {
    var response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"weights": weights}),
    );

    if (response.statusCode == 200) {
      print("✅ Model update sent successfully");
      return true;
    } else {
      print("❌ Failed to send model update: ${response.statusCode}");
      return false;
    }
  } catch (e) {
    print("❌ Error sending model update: $e");
    return false;
  }
}

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Federated SLM App',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: MyModelPage(),
    );
  }
}

class MyModelPage extends StatefulWidget {
  @override
  _MyModelPageState createState() => _MyModelPageState();
}

class _MyModelPageState extends State<MyModelPage> {
  TensorFlowLiteModel model = TensorFlowLiteModel();
  bool _isLoading = false;  // Track loading state
  String _statusMessage = '';  // To show the status message

  @override
  void initState() {
    super.initState();
    model.loadModel();  // Load the model when the page is initialized
  }

  @override
  void dispose() {
    model.close();  // Close the model when the page is disposed
    super.dispose();
  }

  // Function to run the model and send the weights to the server
  Future<void> makePrediction() async {
    if (_isLoading) return;  // Prevent multiple clicks

    setState(() {
      _isLoading = true;  // Show loading indicator
      _statusMessage = 'Running inference...';  // Show status message
    });

    try {
      double input = 1.23;  // Example input, replace with actual input value
      var result = await model.runModel(input);  // Run the model

      if (result != null) {
        List<double> weights = result.cast<double>();  // Assuming 'result' is a list of weights
        bool success = await sendModelWeights(weights);  // Send weights to the server

        setState(() {
          _statusMessage = success ? '✅ Model updated successfully!' : '❌ Failed to update model';
        });
      } else {
        setState(() {
          _statusMessage = '⚠️ Inference failed. Try again.';
        });
      }
    } catch (e) {
      setState(() {
        _statusMessage = '❌ Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;  // Hide loading indicator
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Federated Model Inference")),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: _isLoading ? null : makePrediction, // Disable button while loading
                child: _isLoading ? Text("Processing...") : Text("Run Inference"),
              ),
              SizedBox(height: 20),
              if (_isLoading) CircularProgressIndicator(),
              if (_statusMessage.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 10.0),
                  child: Text(_statusMessage, style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
