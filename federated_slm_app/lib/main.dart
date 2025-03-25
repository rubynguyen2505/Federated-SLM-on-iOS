import 'package:flutter/material.dart';
import 'tflite_model.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Federated Learning Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: FederatedLearningDemo(),
    );
  }
}

class FederatedLearningDemo extends StatefulWidget {
  @override
  _FederatedLearningDemoState createState() => _FederatedLearningDemoState();
}

class _FederatedLearningDemoState extends State<FederatedLearningDemo> {
  final TensorFlowLiteModel _model = TensorFlowLiteModel();
  String _status = "Loading...";

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  // Load the model when the app starts
  void loadModel() async {
    String result = await _model.loadModel();
    setState(() {
      _status = result;
    });
  }

  // Simulate federated learning process
  void simulateFederatedLearning() async {
    // Simulate local model update
    _model.simulateLocalUpdate();

    // Send model weights to the server for aggregation
    String sendResponse = await _model.sendWeightsToServer(_model.modelWeights!);

    // Receive aggregated model from the server
    String receiveResponse = await _model.receiveAggregatedModel();

    setState(() {
      _status = "Federated learning demo completed: $sendResponse, $receiveResponse";
    });
  }

  @override
  void dispose() {
    _model.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Federated Learning Demo'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(_status),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: simulateFederatedLearning,
              child: Text('Start Federated Learning Simulation'),
            ),
          ],
        ),
      ),
    );
  }
}
