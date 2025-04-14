import 'package:flutter/material.dart';
import 'tflite_model.dart';
import 'dart:math';

void main() async {
  // Ensure Flutter engine is fully initialized before any platform channel calls
  WidgetsFlutterBinding.ensureInitialized();
  
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
  String _aggregatedWeights = "";

  @override
  void initState() {
    super.initState();

    // Delayed model loading after the widget is initialized
    WidgetsBinding.instance.addPostFrameCallback((_) {
      loadModel();
    });
  }

  void loadModel() async {
    String result = await _model.loadModel();
    setState(() {
      _status = result;
    });
  }

  void federatedLearning() async {
    for (int round = 1; round <= 100; round++) {
      print("Round $round: Starting federated learning update...");

      _model.localUpdate();

      String sendResponse = await _model.sendWeightsToServer(_model.modelWeights!);
      String receiveResponse = await _model.receiveAggregatedModel();

      setState(() {
        _status = "Round $round: $sendResponse, $receiveResponse";
        _aggregatedWeights = receiveResponse;
      });

      Random random = Random();
      await Future.delayed(Duration(seconds: random.nextInt(8) + 1));
    }
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
            Text(
              _status,
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: federatedLearning,
              child: Text('Start Federated Learning Simulation'),
            ),
            SizedBox(height: 20),
            ExpansionTile(
              title: Text('Aggregated Model Weights'),
              children: [
                Container(
                  height: MediaQuery.of(context).size.height * 0.2,
                  width: MediaQuery.of(context).size.width * 0.8,
                  padding: EdgeInsets.all(8.0),
                  child: Scrollbar(
                    thumbVisibility: true,
                    child: SingleChildScrollView(
                      child: ConstrainedBox(
                        constraints: BoxConstraints(
                          maxHeight: MediaQuery.of(context).size.height * 0.2,
                        ),
                        child: SelectableText(
                          _aggregatedWeights.isEmpty
                              ? 'No aggregated weights yet.'
                              : _aggregatedWeights,
                          style: TextStyle(fontSize: 12),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
