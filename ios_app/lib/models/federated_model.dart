import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:tflite_flutter/tflite_flutter.dart';

class FederatedModel {
  late Interpreter interpreter;

  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset('assets/model.tflite');
  }

  Future<List<double>> runInference(List<int> input) async {
    // Convert input to Float32List for TensorFlow Lite
    var inputTensor = Float32List.fromList(input.map((e) => e.toDouble()).toList());

    // Create an output buffer based on the model’s output shape
    var outputTensor = List.filled(10, 0.0).reshape([1, 10]);

    // Run inference
    interpreter.run(inputTensor.reshape([1, input.length]), outputTensor);

    return List<double>.from(outputTensor[0]);
  }

  // Send the model weights to the server
  Future<void> sendModelUpdate(List<double> weights) async {
    final url = Uri.parse('http://your-server-ip:5000/upload_model');
    final headers = {"Content-Type": "application/json"};
    final body = json.encode({"weights": weights});

    final response = await http.post(url, headers: headers, body: body);
    if (response.statusCode == 200) {
      print('Model update sent successfully!');
    } else {
      print('Failed to send model update.');
    }
  }

  // Extract model weights and send to the server
  Future<void> sendWeights() async {
    List<double> weights = getWeightsFromInterpreter();
    await sendModelUpdate(weights);
  }

  // Extract weights from the model
  List<double> getWeightsFromInterpreter() {
    var tensorShape = interpreter.getOutputTensor(0).shape;
    var buffer = Float32List(tensorShape.reduce((a, b) => a * b)); // Flatten shape
    interpreter.getOutputTensor(0).copyTo(buffer);
    return buffer.toList();
  }
}
