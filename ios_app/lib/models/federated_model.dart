import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:tflite_flutter/tflite_flutter.dart';

class FederatedModel {
  late Interpreter interpreter;
  bool _isModelLoaded = false; // ✅ Flag to track if the model is loaded

  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset('assets/model.tflite');
    _isModelLoaded = true; // ✅ Mark model as loaded
    print("✅ Model loaded successfully!");
  }

  Future<List<double>> runInference(List<int> input) async {
    if (!_isModelLoaded) {
      throw Exception("⚠️ Model not loaded! Call loadModel() first.");
    }

    // ✅ Convert input to Float32List for TensorFlow Lite
    var inputTensor = Float32List.fromList(input.map((e) => e.toDouble()).toList());

    // ✅ Create an output buffer correctly
    var outputTensor = Float32List(10); // Adjust this size based on your model

    // ✅ Run inference
    interpreter.run(inputTensor.reshape([1, input.length]), outputTensor);

    return outputTensor.toList();
  }

  // ✅ Send model weights to the server
  Future<void> sendModelUpdate(List<double> weights) async {
    final url = Uri.parse('http://your-server-ip:5000/upload_model');
    final headers = {"Content-Type": "application/json"};
    final body = json.encode({"weights": weights});

    final response = await http.post(url, headers: headers, body: body);
    if (response.statusCode == 200) {
      print('✅ Model update sent successfully!');
    } else {
      print('❌ Failed to send model update.');
    }
  }

  // ✅ Extract model weights and send to the server
  Future<void> sendWeights() async {
    if (!_isModelLoaded) {
      throw Exception("⚠️ Model not loaded! Call loadModel() first.");
    }

    List<double> weights = getWeightsFromInterpreter();
    await sendModelUpdate(weights);
  }

  // ✅ Extract weights correctly
  List<double> getWeightsFromInterpreter() {
    var outputTensor = interpreter.getOutputTensor(0);
    var buffer = outputTensor.data.buffer.asFloat32List(); // ✅ Correct way
    return buffer.toList();
  }
}
