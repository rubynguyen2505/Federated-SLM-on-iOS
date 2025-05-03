import UIKit
import Flutter
import CoreML
import ZIPFoundation

@main
@objc class AppDelegate: FlutterAppDelegate {
  var tokenizer: [String: Any]? = nil
  var currentModel: imdb_updatable_model? = nil
  var currentModelVersion: String? = nil

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    let controller = window.rootViewController as! FlutterViewController
    let channel = FlutterMethodChannel(name: "com.example.coreml", binaryMessenger: controller.binaryMessenger)

    // Load tokenizer
    if let tokenizerURL = Bundle.main.url(forResource: "tokenizer.json", withExtension: nil),
       let data = try? Data(contentsOf: tokenizerURL),
       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
      tokenizer = json
    }

    // Load default model at app launch
    do {
      let defaultModel = try imdb_updatable_model(configuration: MLModelConfiguration())
      self.currentModel = defaultModel
      print("✅ Default model loaded at launch")
    } catch {
      print("❌ Failed to load default model: \(error)")
    }

    channel.setMethodCallHandler { [weak self] call, result in
      guard let self = self else { return }

      switch call.method {
      case "predict":
        self.handlePrediction(call: call, result: result)
      case "train":
        self.handleTraining(call: call, result: result)
      case "download":
        self.downloadAggregatedModel(call: call, result: result)
      default:
        result(FlutterMethodNotImplemented)
      }
    }

    GeneratedPluginRegistrant.register(with: self)
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func tokenize(_ text: String, wordIndex: [String: Int]) -> [Int] {
    let tokens = text
      .lowercased()
      .components(separatedBy: .whitespacesAndNewlines)
      .compactMap { wordIndex[$0] }

    let padded = Array(tokens.prefix(100)) + Array(repeating: 0, count: max(0, 100 - tokens.count))
    return padded
  }

  func handlePrediction(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let text = args["text"] as? String,
          let tokenizer = tokenizer,
          let wordIndex = tokenizer["word_index"] as? [String: Int],
          let model = currentModel else {
      result(FlutterError(code: "INVALID_ARGUMENT", message: "Missing text/tokenizer/model", details: nil))
      return
    }

    do {
      let tokens = tokenize(text, wordIndex: wordIndex)
      let input = try MLMultiArray(shape: [1, 100], dataType: .int32)
      for (i, token) in tokens.enumerated() {
        input[[0, NSNumber(value: i)]] = NSNumber(value: token)
      }

      let output = try model.prediction(embedding_input: input)
      let probs = output.Identity
      result((0..<probs.count).map { probs[$0].doubleValue })

    } catch {
      result(FlutterError(code: "PREDICTION_FAILED", message: error.localizedDescription, details: nil))
    }
  }

  func handleTraining(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any],
          let samples = args["examples"] as? [[String: Any]],
          let tokenizer = tokenizer,
          let wordIndex = tokenizer["word_index"] as? [String: Int],
          let _ = currentModel else {
      result(FlutterError(code: "INVALID_ARGUMENT", message: "Missing training data/model", details: nil))
      return
    }

    do {
      var featureProviders: [MLFeatureProvider] = []

      // Prepare training data
      for sample in samples {
        guard let text = sample["text"] as? String,
              let label = sample["label"] as? Int else {
          continue
        }

        let tokens = tokenize(text, wordIndex: wordIndex)
        let inputArray = try MLMultiArray(shape: [1, 100], dataType: .int32)
        for (i, token) in tokens.enumerated() {
          inputArray[[0, NSNumber(value: i)]] = NSNumber(value: token)
        }

        let labelArray = try MLMultiArray(shape: [1], dataType: .int32)
        labelArray[0] = NSNumber(value: label)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
          "embedding_input": MLFeatureValue(multiArray: inputArray),
          "Identity_true": MLFeatureValue(multiArray: labelArray)
        ])

        featureProviders.append(provider)
      }

      guard let modelURL = Bundle.main.url(forResource: "imdb_updatable_model", withExtension: "mlmodelc") else {
        result(FlutterError(code: "MODEL_NOT_FOUND", message: "Model URL missing", details: nil))
        return
      }

      let config = MLModelConfiguration()
      let batchProvider = MLArrayBatchProvider(array: featureProviders)

      let task = try MLUpdateTask(forModelAt: modelURL,
                                  trainingData: batchProvider,
                                  configuration: config,
                                  progressHandlers: MLUpdateProgressHandlers(
                                    forEvents: [.trainingBegin, .epochEnd],
                                    progressHandler: { context in
                                      print("Training progress: epoch \(context.metrics[.epochIndex] ?? "?"), loss \(context.metrics[.lossValue] ?? "?")")
                                    },
                                    completionHandler: { context in
                                      if context.task.state == .completed {
                                        var correctPredictions = 0
                                        var totalPredictions = 0

                                        for sample in samples {
                                          guard let text = sample["text"] as? String,
                                                let label = sample["label"] as? Int,
                                                let trainedModel = self.currentModel else {
                                            result(FlutterError(code: "INVALID_ARGUMENT", message: "Missing text/tokenizer/model", details: nil))
                                            return
                                          }

                                          do {
                                            let tokens = self.tokenize(text, wordIndex: wordIndex)
                                            let input = try MLMultiArray(shape: [1, 100], dataType: .int32)
                                            for (i, token) in tokens.enumerated() {
                                              input[[0, NSNumber(value: i)]] = NSNumber(value: token)
                                            }

                                            let output = try trainedModel.prediction(embedding_input: input)
                                            let probs = output.Identity
                                            if probs.count == 2 {
                                              // The second index corresponds to the probability of positive sentiment
                                              let predictedLabel = probs[1].doubleValue > 0.5 ? 1 : 0

                                              // Compare with actual label and calculate accuracy
                                              if predictedLabel == label {
                                                correctPredictions += 1
                                              }
                                              totalPredictions += 1
                                            }
                                          } catch {
                                            result(FlutterError(code: "PREDICTION_FAILED", message: error.localizedDescription, details: nil))
                                          }
                                        }

                                        let accuracy = Double(correctPredictions) / Double(totalPredictions) * 100
                                        print("Training accuracy: \(accuracy)%")

                                        let loss = context.metrics[.lossValue] as? Double ?? 0.0

                                        // Send metrics to server (optional)
                                        self.sendMetricsToServer(accuracy: accuracy, loss: loss, modelVersion: self.currentModelVersion ?? "Unknown")

                                        let updatedModelURL = FileManager.default.temporaryDirectory.appendingPathComponent("updated_model.mlmodelc")
                                        do {
                                          try context.model.write(to: updatedModelURL)
                                          self.uploadModel(to: updatedModelURL)
                                          result("Training accuracy: \(accuracy)%")
                                        } catch {
                                          result(FlutterError(code: "SAVE_FAILED", message: error.localizedDescription, details: nil))
                                        }
                                      } else {
                                        result(FlutterError(code: "TRAINING_FAILED", message: "Task did not complete", details: nil))
                                      }
                                    }
                                  ))
      task.resume()
    } catch {
      result(FlutterError(code: "TRAINING_ERROR", message: error.localizedDescription, details: nil))
    }
  }

  func uploadModel(to url: URL) {
    let zipURL = FileManager.default.temporaryDirectory.appendingPathComponent("updated_model.zip")
    do {
      if FileManager.default.fileExists(atPath: zipURL.path) {
        try FileManager.default.removeItem(at: zipURL)
      }
      try FileManager.default.zipItem(at: url, to: zipURL)
      var request = URLRequest(url: URL(string: "http://192.168.12.118:5000/upload")!)
      request.httpMethod = "POST"
      let boundary = UUID().uuidString
      request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

      let data = try Data(contentsOf: zipURL)
      var body = Data()
      body.append("--\(boundary)\r\n".data(using: .utf8)!)
      body.append("Content-Disposition: form-data; name=\"model\"; filename=\"updated_model.zip\"\r\n".data(using: .utf8)!)
      body.append("Content-Type: application/octet-stream\r\n\r\n".data(using: .utf8)!)
      body.append(data)
      body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
      request.httpBody = body

      let task = URLSession.shared.dataTask(with: request) { _, _, error in
        if let error = error {
          print("❌ Upload failed: \(error)")
          return
        }
        print("✅ Upload successful")
        self.callAggregationEndpoint()
      }

      task.resume()
    } catch {
      print("Zipping or uploading failed \(error)")
    }
  }

  func callAggregationEndpoint() {
    let aggregationURL = URL(string: "http://192.168.12.118:5000/aggregate")!
    var request = URLRequest(url: aggregationURL)
    request.httpMethod = "POST"

    let task = URLSession.shared.dataTask(with: request) { data, _, error in
      if let error = error {
        print("❌ Aggregation failed: \(error)")
        return
      }
      if let data = data, let responseString = String(data: data, encoding: .utf8) {
        print("✅ Aggregation successful: \(responseString)")
      }
    }
    task.resume()
  }

  func downloadAggregatedModel(call: FlutterMethodCall, result: @escaping FlutterResult) {
    let downloadURL = URL(string: "http://192.168.12.118:5000/download")!
    let task = URLSession.shared.dataTask(with: downloadURL) { data, response, error in
      if let error = error {
        print("❌ Download failed: \(error)")
        return
      }

      if let httpResponse = response as? HTTPURLResponse {
        // Extract the model version from the response headers
        if let modelVersion = httpResponse.allHeaderFields["Model-Version"] as? String {
          self.currentModelVersion = modelVersion
          print("✅ Model Version: \(modelVersion)")
        } else {
          print("❌ Model version not found in response headers")
        }
      }

      if let data = data {
        let fileURL = FileManager.default.temporaryDirectory.appendingPathComponent("aggregated_model.mlmodel")
        do {
          try data.write(to: fileURL)
          print("✅ Aggregated model downloaded at \(fileURL)")
          self.convertAndLoadModel(from: fileURL)
        } catch {
          print("❌ Saving downloaded model failed: \(error)")
        }
      }

      result("Downloaded successfully.")
    }
    task.resume()
  }

  func convertAndLoadModel(from fileURL: URL) {
    do {
      let compiledModelURL = try MLModel.compileModel(at: fileURL)
      print("✅ Model compiled and saved to: \(compiledModelURL)")
      self.loadUpdatedModel(from: compiledModelURL)
    } catch {
      print("❌ Model conversion failed: \(error)")
    }
  }

  func loadUpdatedModel(from fileURL: URL) {
    do {
      let model = try imdb_updatable_model(contentsOf: fileURL, configuration: MLModelConfiguration())
      self.currentModel = model
      print("✅ Updated model loaded successfully and now in use")
    } catch {
      print("❌ Loading updated model failed: \(error)")
    }
  }

  func sendMetricsToServer(accuracy: Double, loss: Double, modelVersion: String) {
    let metricsURL = URL(string: "http://192.168.12.118:5000/metrics")!
    var request = URLRequest(url: metricsURL)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    // Prepare the metrics payload
    let metricsPayload: [String: Any] = [
        "accuracy": accuracy,
        "loss": loss,
        "model_version": modelVersion
    ]
    
    do {
        let jsonData = try JSONSerialization.data(withJSONObject: metricsPayload, options: [])
        request.httpBody = jsonData
        
        // Send the request
        let task = URLSession.shared.dataTask(with: request) { data, _, error in
            if let error = error {
                print("❌ Failed to send metrics: \(error)")
                return
            }
            print("✅ Metrics sent successfully!")
        }
        task.resume()
    } catch {
        print("❌ Failed to serialize metrics: \(error)")
    }
  }

}
