import 'package:tflite/tflite.dart';
import 'dart:typed_data';  // Import for Uint8List

class TensorFlowLiteModel {
  // Initialize the model
  Future<void> loadModel() async {
    String? res = await Tflite.loadModel(
      model: 'assets/model.tflite',
      labels: 'assets/labels.txt', // Optionally, provide labels if available
    );

    if (res == null) {
      print("❌ Model failed to load!");
    } else {
      print("✅ Model loaded successfully: $res");
    }
  }

  // Run inference on an input tensor (expecting a single float value)
  Future<List<dynamic>?> runModel(double input) async {
    try {
      // Convert the input to binary format
      // Ensure you create a Uint8List from the input (typically needed for binary data)
      Uint8List inputData = Uint8List(1)..buffer.asByteData().setFloat32(0, input, Endian.little); // Convert float to binary

      // Run inference with the binary input
      var output = await Tflite.runModelOnBinary(
        binary: inputData, // Use Uint8List as input
        numResults: 2,  
        threshold: 0.05,  
        asynch: true,  
      );

      print("✅ Inference output: $output");
      return output;
    } catch (e) {
      print("❌ Error during inference: $e");
      return null;
    }
  }

  // Close the model when done
  void close() {
    Tflite.close();
  }
}
