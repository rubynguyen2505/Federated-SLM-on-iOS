import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:federated_slm_app/tflite_model.dart';
import 'package:http/http.dart' as http;

// Function to send model weights to the server
Future<void> sendModelWeights(List<double> weights) async {
  var url = Uri.parse("http://<YOUR_SERVER_IP>:5000/upload_model");

  // Send HTTP POST request with weights
  var response = await http.post(
    url,
    headers: {"Content-Type": "application/json"},
    body: jsonEncode({"weights": weights}),
  );

  if (response.statusCode == 200) {
    print("✅ Model update sent successfully");
  } else {
    print("❌ Failed to send model update");
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
  void makePrediction() async {
    double input = 1.23;  // Example input, replace with actual input value
    var result = await model.runModel(input);  // Run the model

    if (result != null) {
      List<double> weights = result.cast<double>();  // Assuming 'result' is a list of weights
      await sendModelWeights(weights);  // Send weights to the server
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Federated Model Inference")),
      body: Center(
        child: ElevatedButton(
          onPressed: () => makePrediction(),
          child: Text("Run Inference"),
        ),
      ),
    );
  }
}
