import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class TensorFlowLiteModel {
  Interpreter? _interpreter;
  Map<String, dynamic>? _tokenizer; // Tokenizer data

  // Load model and tokenizer
  Future<String> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      
      // Load tokenizer
      String jsonString = await rootBundle.loadString('assets/tokenizer.json');
      _tokenizer = jsonDecode(jsonString);
      
      if (_interpreter == null) {
        print("❌ Model or Tokenizer not loaded!");
        return "❌ Model or Tokenizer not loaded!";
      } else {
        print("✅ Model and Tokenizer loaded successfully!");
        return "✅ Model and Tokenizer loaded successfully!";
      }
    } catch (e) {
      print("❌ Error loading model/tokenizer: $e");
      return "❌ Error loading model/tokenizer: $e";
    }
  }

  // Tokenize and pad input text
  List<int> tokenizeAndPad(String text, int maxLen) {
    if (_tokenizer == null) {
      print("❌ Tokenizer not loaded!");
      return [];
    }

    // Tokenizer word index
    Map<String, dynamic> wordIndex = _tokenizer!['config']['word_index'];

    // Convert text to tokenized sequence (cast values to int explicitly)
    List<int> sequence = text
        .split(" ")
        .map((word) => (wordIndex[word] ?? 0) as int) // 👈 Explicitly cast to int
        .toList();

    // Pad sequence to max length
    List<int> paddedSequence = List.filled(maxLen, 0);
    for (int i = 0; i < sequence.length && i < maxLen; i++) {
      paddedSequence[i] = sequence[i];
    }

    return paddedSequence;
  }

  // Run model inference
  Future<List<dynamic>?> runModel(String textInput) async {
    try {
      List<int> inputData = tokenizeAndPad(textInput, 100); // Tokenize & pad
      
      // Model expects a batch of inputs
      List<List<int>> modelInput = [inputData];

      var output = List.filled(10, 0.0); // Adjust output size if needed
      _interpreter?.run(modelInput, output);

      print("✅ Inference output: $output");
      return output;
    } catch (e) {
      print("❌ Error during inference: $e");
      return null;
    }
  }

  void close() {
    _interpreter?.close();
  }
}
