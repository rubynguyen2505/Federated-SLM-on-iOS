class MyModelPage extends StatefulWidget {
  @override
  _MyModelPageState createState() => _MyModelPageState();
}

class _MyModelPageState extends State<MyModelPage> {
  TensorFlowLiteModel model = TensorFlowLiteModel();

  @override
  void initState() {
    super.initState();
    model.loadModel();  // Load the model when the page is initialized
  }

  @override
  void dispose() {
    model.close();  // Close the model when the page is disposed
    super.dispose();
  }

  void makePrediction() async {
    // Example input, adjust based on your model
    var input = [/* input data */];
    var result = await model.runModel(input);
    print(result);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Federated Model Inference")),
      body: Center(
        child: ElevatedButton(
          onPressed: makePrediction,
          child: Text("Run Inference"),
        ),
      ),
    );
  }
}
