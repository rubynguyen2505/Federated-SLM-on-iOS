import 'dart:convert';
import 'package:http/http.dart' as http;

class FederatedModel {
  late Interpreter interpreter;

  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset('assets/model.tflite');
  }

  Future<List<dynamic>> runInference(List<int> input) async {
    var inputTensor = Tensor.fromList(input);
    var output = List.generate(10, (index) => 0.0);
    interpreter.run(inputTensor, output);
    return output;
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
    // Placeholder function for extracting weights from the interpreter
    // You'll need to extract the actual weights from the model here
    return [0.1, 0.2, 0.3];  // Example weights, replace with real ones
  }
}
