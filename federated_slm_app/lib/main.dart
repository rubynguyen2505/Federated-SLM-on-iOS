import 'package:flutter/material.dart';
import 'package:federated_slm_app/tflite_model.dart';

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

  void makePrediction() async {
    // Example input - this should be a single double value based on your model's input
    double input = 1.23;  // Replace with actual input value
    var result = await model.runModel(input);  // Pass the double directly to the runModel function
    print(result);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Federated Model Inference")),
      body: Center(
        child: ElevatedButton(
          onPressed: makePrediction,
          child: Text("Run Inference"),
        ),
      ),
    );
  }
}
