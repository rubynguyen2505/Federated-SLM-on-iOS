from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Store model weights from clients
model_weights = []

@app.route("/upload_model", methods=["POST"])
def upload_model():
    global model_weights

    # Get weights from request
    try:
        weights = np.array(request.json["weights"])
        model_weights.append(weights)
        
        # Aggregate weights (average for simplicity)
        aggregated_weights = np.mean(model_weights, axis=0)
        print(f"Aggregated weights: {aggregated_weights}")
        
        return jsonify({"status": "Model update received!", "aggregated_weights": aggregated_weights.tolist()})
    except Exception as e:
        return jsonify({"status": "Error processing weights", "error": str(e)}), 400

@app.route("/get_aggregated_model", methods=["GET"])
def get_aggregated_model():
    if not model_weights:
        return jsonify({"status": "No model weights received yet."}), 400

    # Aggregate weights and send back
    aggregated_weights = np.mean(model_weights, axis=0)
    return jsonify({"status": "Aggregated model sent", "aggregated_weights": aggregated_weights.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
