import 'package:flutter/material.dart';
import 'package:ios_app/models/federated_model.dart'; // Import your model

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final FederatedModel model = FederatedModel();
  String inferenceResult = "Press the button to run inference";

  @override
  void initState() {
    super.initState();
    model.loadModel();
  }

  Future<void> _runInference() async {
    List<int> sampleInput = [1, 2, 3, 4, 5]; // Example input
    List<double> result = await model.runInference(sampleInput);

    setState(() {
      inferenceResult = "Inference Result: ${result.join(", ")}";
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text("Federated SLM on iOS")),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(inferenceResult, textAlign: TextAlign.center),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: _runInference,
                child: Text("Run Inference"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
