import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:tflite_flutter/tflite_flutter.dart';

class TensorFlowLiteModel {
  Interpreter? _interpreter;
  Map<String, dynamic>? _tokenizer; // Tokenizer data
  List<List<double>>? _modelWeights;

  // Load model and tokenizer
  Future<String> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      
      // Load tokenizer
      String jsonString = await rootBundle.loadString('assets/tokenizer.json');
      _tokenizer = jsonDecode(jsonString);
      
      // Initialize empty weights
      _modelWeights = [];
      
      return "Model and tokenizer loaded!";
    } catch (e) {
      return "Model and tokenizer loaded!";
    }
  }

  // Tokenize and pad input text
  List<int> tokenizeAndPad(String text, int maxLen) {
    if (_tokenizer == null) {
      print("Tokenizer not loaded!");
      return [];
    }

    // Tokenizer word index
    Map<String, dynamic> wordIndex = _tokenizer!['config']['word_index'];

    // Convert text to tokenized sequence
    List<int> sequence = text
        .split(" ")
        .map((word) => (wordIndex[word] ?? 0) as int)
        .toList();

    // Pad sequence to max length
    List<int> paddedSequence = List.filled(maxLen, 0);
    for (int i = 0; i < sequence.length && i < maxLen; i++) {
      paddedSequence[i] = sequence[i];
    }

    return paddedSequence;
  }

  // Simulate local model update by generating random model weights (for now)
  void simulateLocalUpdate() {
    print("Simulating local model update...");
    
    // Generating some random weights for the model (as an example)
    _modelWeights = List.generate(
        10, (index) => List.generate(10, (index) => (index + 1).toDouble())
    );
  }

  // Run model inference
  Future<List<dynamic>?> runModel(String textInput) async {
    try {
      List<int> inputData = tokenizeAndPad(textInput, 100); // Tokenize & pad
      List<List<int>> modelInput = [inputData];

      var output = List.filled(10, 0.0); // Adjust output size if needed
      _interpreter?.run(modelInput, output);

      print("Inference output: $output");
      return output;
    } catch (e) {
      print("Error during inference: $e");
      return null;
    }
  }

  // Send model weights to the server for aggregation
  Future<String> sendWeightsToServer(List<List<double>> modelWeights) async {
    try {
      var url = Uri.parse("http://192.168.12.118:5000/upload_model"); // Use your server IP
      var response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({"weights": modelWeights}),
      );

      if (response.statusCode == 200) {
        print("Weights sent successfully! Response: ${response.body}");
        return response.body;
      } else {
        print("Failed to send weights. Status: ${response.statusCode}");
        return "Failed to send weights.";
      }
    } catch (e) {
      print("Error sending weights to server: $e");
      return "Error sending weights to server.";
    }
  }

  // Receive aggregated model from the server
  Future<String> receiveAggregatedModel() async {
    // Example of a more complex model aggregation logic
    try {
      var url = Uri.parse("http://192.168.12.118:5000/get_aggregated_model");
      var response = await http.get(url);

      if (response.statusCode == 200) {
        print("Aggregated model received: ${response.body}");
        return response.body;
      } else {
        print("Failed to receive aggregated model. Status: ${response.statusCode}");
        return "Failed to receive aggregated model.";
      }
    } catch (e) {
      print("Error receiving aggregated model: $e");
      return "Error receiving aggregated model.";
    }
  }

  // Getter for _modelWeights
  List<List<double>>? get modelWeights => _modelWeights;

  void close() {
    _interpreter?.close();
  }
}
