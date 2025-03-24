import 'package:tflite_flutter/tflite_flutter.dart';

class TensorFlowLiteModel {
  Interpreter? _interpreter;

  // Initialize the model
  Future<void> loadModel() async {
    try {
      // Load the model using tflite_flutter's Interpreter
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      print("✅ Model loaded successfully");
    } catch (e) {
      print("❌ Model failed to load: $e");
    }
  }

  // Run inference on an input tensor (expecting a single float value)
  Future<List<dynamic>?> runModel(double input) async {
    try {
      // Prepare input data as a List<List<double>> for a 2D input tensor
      List<List<double>> inputData = [[input]];

      // Create an output buffer to hold the result
      var output = List.filled(2, 0.0);  // Change this based on your model's expected output size

      // Run inference on the input data
      _interpreter?.run(inputData, output);

      print("✅ Inference output: $output");
      return output;
    } catch (e) {
      print("❌ Error during inference: $e");
      return null;
    }
  }

  // Close the model when done
  void close() {
    _interpreter?.close();
  }
}
