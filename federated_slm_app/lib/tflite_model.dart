import 'package:tflite/tflite.dart';

class TensorFlowLiteModel {
  // Initialize the model
  Future<void> loadModel() async {
    await Tflite.loadModel(
      model: 'assets/model.tflite',
      labels: 'assets/labels.txt', // Optionally, provide labels if available
    );
  }

  // Run inference on an input tensor
  Future<List<dynamic>> runModel(List<dynamic> input) async {
    var output = await Tflite.runModelOnArray(
      input,  // Input tensor (adjust depending on the model's input type)
      numResults: 2,  // Number of output results (adjust as needed)
    );
    return output;
  }

  // Close the model when done
  void close() {
    Tflite.close();
  }
}
