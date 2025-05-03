import 'dart:convert';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Core ML Sentiment Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const MyHomePage(title: 'Core ML Sentiment Analysis'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String _modelOutput = 'Waiting for action...';

  static const platform = MethodChannel('com.example.coreml');

  List<Map<String, dynamic>> _serverTestExamples = [];

  final TextEditingController _predictTextController = TextEditingController();

  bool _isFederatedLearningRunning = false;

  final TextEditingController _roundsController = TextEditingController(text: '1');
  int _numRounds = 1;

  Future<void> _fetchAndTrainFromServer() async {
    const String serverUrl = 'http://192.168.12.118:5000/get_train_data';

    setState(() => _modelOutput = 'Fetching validation data...');

    try {
      final response = await http.get(Uri.parse(serverUrl));

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseBody = json.decode(response.body);
        final List<dynamic> data = responseBody['data'];

        final examples = data
            .map<Map<String, dynamic>>((e) => {
                  'text': e['text'],
                  'label': e['label'],
                })
            .toList();

        final String result = await platform.invokeMethod('train', {
          'examples': examples,
        });

        setState(() => _modelOutput = 'Training complete: $result');
      } else {
        setState(() => _modelOutput = 'Server Error: ${response.statusCode}');
      }
    } catch (e) {
      setState(() => _modelOutput = 'Training Error: $e');
    }
  }

  Future<void> _fetchAndPredictFromServer({required int currentRound}) async {
    const String serverUrl = 'http://192.168.12.118:5000/get_test_data_gzip';
    const String metricsUploadUrl = 'http://192.168.12.118:5000/report_metrics';
    const String clientId = 'flutter_client';

    setState(() => _modelOutput = 'Fetching test data...');

    try {
      final response = await http.get(Uri.parse(serverUrl));

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = json.decode(response.body);

        if (data['data'] is! List) {
          setState(() => _modelOutput = 'Unexpected data format from server.');
          return;
        }

        _serverTestExamples = (data['data'] as List)
            .map<Map<String, dynamic>>((e) => {
                  'text': e['text'],
                  'label': e['label'],
                })
            .toList();

        final int total = _serverTestExamples.length;
        int tp = 0, tn = 0, fp = 0, fn = 0;
        List<double> predictedProbs = [];
        Map<int, int> classTP = {0: 0, 1: 0};
        Map<int, int> classFP = {0: 0, 1: 0};
        Map<int, int> classFN = {0: 0, 1: 0};
        Map<int, int> classTN = {0: 0, 1: 0};

        final stopwatch = Stopwatch()..start();

        for (var sample in _serverTestExamples) {
          final List<dynamic> result = await platform.invokeMethod('predict', {
            'text': sample['text'],
          });

          double negativeScore = result[0];
          double positiveScore = result[1];
          int predicted = positiveScore >= negativeScore ? 1 : 0;
          int actual = sample['label'];

          // Update confusion matrix
          if (predicted == 1 && actual == 1) {
            tp++;
            classTP[1] = classTP[1]! + 1;
          } else if (predicted == 0 && actual == 0) {
            tn++;
            classTN[0] = classTN[0]! + 1;
          } else if (predicted == 1 && actual == 0) {
            fp++;
            classFP[0] = classFP[0]! + 1;
          } else if (predicted == 0 && actual == 1) {
            fn++;
            classFN[1] = classFN[1]! + 1;
          }

          predictedProbs.add(positiveScore);
        }

        final double accuracy = total == 0 ? 0 : (tp + tn) / total;
        final double precision = (tp + fp) == 0 ? 0 : tp / (tp + fp);
        final double recall = (tp + fn) == 0 ? 0 : tp / (tp + fn);
        final double f1 = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);

        final double logLoss = -predictedProbs.asMap().map((i, p) {
          double y = _serverTestExamples[i]['label'] == 1 ? 1.0 : 0.0;
          p = p.clamp(1e-15, 1 - 1e-15);
          return MapEntry(i, y * log(p) + (1 - y) * log(1 - p));
        }).values.reduce((a, b) => a + b) / total;

        final stopwatchDuration = stopwatch.elapsedMilliseconds;

        // Per-class Precision and Recall
        final double precision0 = classTP[0]! + classFP[0]! == 0
            ? 0
            : classTP[0]! / (classTP[0]! + classFP[0]!);
        final double recall0 = classTP[0]! + classFN[0]! == 0
            ? 0
            : classTP[0]! / (classTP[0]! + classFN[0]!);

        final double precision1 = classTP[1]! + classFP[1]! == 0
            ? 0
            : classTP[1]! / (classTP[1]! + classFP[1]!);
        final double recall1 = classTP[1]! + classFN[1]! == 0
            ? 0
            : classTP[1]! / (classTP[1]! + classFN[1]!);

        // Update UI
        setState(() {
          _modelOutput = 'Prediction complete.\n'
              'Accuracy: ${(accuracy * 100).toStringAsFixed(2)}%\n'
              'Precision: ${(precision * 100).toStringAsFixed(2)}%\n'
              'Recall: ${(recall * 100).toStringAsFixed(2)}%\n'
              'F1 Score: ${(f1 * 100).toStringAsFixed(2)}%';
        });

        final metricsPayload = {
          'client_id': clientId,
          'round': currentRound,
          'accuracy': accuracy,
          'precision': precision,
          'recall': recall,
          'f1_score': f1,
          'log_loss': logLoss,
          'confusion_matrix': {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
          },
          'evaluation_time_ms': stopwatchDuration,
          'prediction_confidence': {
            'average': predictedProbs.reduce((a, b) => a + b) / total,
            'min': predictedProbs.reduce((a, b) => a < b ? a : b),
            'max': predictedProbs.reduce((a, b) => a > b ? a : b),
          },
          'per_class_precision': {
            'class_0': precision0,
            'class_1': precision1,
          },
          'per_class_recall': {
            'class_0': recall0,
            'class_1': recall1,
          },
        };

        final metricsResponse = await http.post(
          Uri.parse(metricsUploadUrl),
          headers: {'Content-Type': 'application/json'},
          body: json.encode(metricsPayload),
        );

        if (metricsResponse.statusCode != 200) {
          print('⚠️ Failed to send metrics: ${metricsResponse.statusCode}');
        } else {
          print('✅ Metrics sent successfully for round $currentRound');
        }
      } else {
        setState(() => _modelOutput = 'Server Error: ${response.statusCode}');
      }
    } catch (e) {
      setState(() => _modelOutput = 'Prediction Error: $e');
    }
  }

  Future<void> _startFederatedLearning() async {
    setState(() {
      _isFederatedLearningRunning = true;
    });

    try {
      _numRounds = int.tryParse(_roundsController.text) ?? 1;

      setState(() {
        _modelOutput = 'Starting federated learning for $_numRounds rounds...';
      });

      for (int round = 1; round <= _numRounds; round++) {
        setState(() {
          _modelOutput = 'Round $round of $_numRounds: Training...';
        });
        await _fetchAndTrainFromServer();

        setState(() {
          _modelOutput = 'Round $round of $_numRounds: Updating model...';
        });
        await updateModel();

        setState(() {
          _modelOutput = 'Round $round of $_numRounds: Predicting...';
        });
        await _fetchAndPredictFromServer(currentRound: round);
        
        setState(() {
          _modelOutput = 'Evaluation completed';
        });
      }

      setState(() {
        _modelOutput = 'Federated learning complete after $_numRounds rounds.';
      });
    } catch (e) {
      setState(() {
        _modelOutput = 'Federated Learning Error: $e';
      });
    } finally {
      setState(() {
        _isFederatedLearningRunning = false;
      });
    }
  }

  Future<void> _manualPredict(String text) async {
    try {
      final List<dynamic> result = await platform.invokeMethod('predict', {
        'text': text,
      });

      double negativeScore = result[0];
      double positiveScore = result[1];

      String predictionLabel =
          positiveScore >= negativeScore ? 'Positive' : 'Negative';

      setState(() {
        _modelOutput =
            'Prediction: $predictionLabel\n(Confidence: Positive ${positiveScore.toStringAsFixed(2)}, Negative ${negativeScore.toStringAsFixed(2)})';
      });
    } catch (e) {
      setState(() {
        _modelOutput = 'Prediction Error: $e';
      });
    }
  }

  Future<void> updateModel() async {
    try {
      final String result = await platform.invokeMethod('download');
      setState(() {
        _modelOutput = 'Model updated successfully! $result';
      });
    } on PlatformException catch (e) {
      setState(() {
        _modelOutput = 'Error while updating model: ${e.message}';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _roundsController,
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Number of Federated Learning Rounds',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 10),
            ElevatedButton(
              onPressed: _isFederatedLearningRunning ? null : _startFederatedLearning,
              style: ElevatedButton.styleFrom(backgroundColor: Colors.purple),
              child: _isFederatedLearningRunning
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Text('Start Federated Learning'),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _predictTextController,
              decoration: const InputDecoration(
                labelText: 'Enter text for manual prediction',
                border: OutlineInputBorder(),
              ),
              maxLines: 2,
            ),
            const SizedBox(height: 10),
            ElevatedButton(
              onPressed: () {
                final text = _predictTextController.text.trim();
                if (text.isNotEmpty) {
                  _manualPredict(text);
                } else {
                  setState(() {
                    _modelOutput = 'Please enter text to predict.';
                  });
                }
              },
              style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
              child: const Text('Manual Predict'),
            ),
            const SizedBox(height: 20),
            Text(
              _modelOutput,
              style: const TextStyle(fontSize: 18),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
