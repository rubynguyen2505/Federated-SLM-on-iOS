import 'package:flutter/material.dart';
import 'federated_model.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: ModelInferenceScreen(),
    );
  }
}

class ModelInferenceScreen extends StatefulWidget {
  @override
  _ModelInferenceScreenState createState() => _ModelInferenceScreenState();
}

class _ModelInferenceScreenState extends State<ModelInferenceScreen> {
  FederatedModel model = FederatedModel();

  @override
  void initState() {
    super.initState();
    // Load the model as soon as the widget is created
    model.loadModel();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Federated Model Inference')),
      body: Center(
        child: ElevatedButton(
          onPressed: () async {
            List<int> inputData = [1, 2, 3, 4];  // Replace with real input data
            var result = await model.runInference(inputData);
            print(result);  // Output the result of the inference
          },
          child: Text('Run Inference'),
        ),
      ),
    );
  }
}
