import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:federated_slm_app/tflite_model.dart';
import 'package:http/http.dart' as http;  // Import the HTTP package

// Function to send model weights to the server
// Function to send model weights to the server
Future<String> sendModelWeights(List<double> weights) async {
  var url = Uri.parse("http://192.168.12.118:5000/upload_model"); // Use Windows Wi-Fi IP

  try {
    print("🔹 Sending model weights to server...");
    print("🔹 Request URL: $url");
    print("🔹 Request Body: ${jsonEncode({"weights": weights})}");

    var response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"weights": weights}),
    );

    print("🔹 Response Status Code: ${response.statusCode}");
    print("🔹 Response Body: ${response.body}");

    if (response.statusCode == 200) {
      print("✅ Model update sent successfully");
      return "✅ Model updated successfully! Server: ${response.body}";
    } else {
      print("❌ Failed to send model update: ${response.statusCode} - ${response.body}");
      return "❌ Server Error (${response.statusCode}): ${response.body}";
    }
  } catch (e) {
    print("❌ Error sending model update: $e");
    return "❌ Network Error: $e";
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
  TextEditingController _textController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    Atring modelLoaded = await model.loadModel();  // Await the asynchronous function
    setState(() {
      _statusMessage = modelLoaded;  // Show the model loading status
    });
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
      _isLoading = true;
      _statusMessage = 'Running inference...';
    });

    try {
      String userInput = _textController.text;
      var result = await model.runModel(userInput);

      if (result != null) {
        setState(() {
          _statusMessage = '✅ Inference successful! Sending model update...';
        });

        List<double> weights = result.cast<double>();  // Ensure it's a list of doubles
        String serverMessage = await sendModelWeights(weights);  // Send weights and get response message

        setState(() {
          _statusMessage = serverMessage;  // Show server response message
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
        _isLoading = false;
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
            children: [
              TextField(
                controller: _textController,
                decoration: InputDecoration(labelText: "Enter text"),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: _isLoading ? null : makePrediction,
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
