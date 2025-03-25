from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Store model weights from clients
model_weights = []
client_data_counts = []  # Track the number of data points from each client for weighted averaging

@app.route("/upload_model", methods=["POST"])
def upload_model():
    global model_weights, client_data_counts

    try:
        # Get weights and data count from the request
        weights = np.array(request.json["weights"])
        data_count = request.json["data_count"]  # Number of data points client has
        
        model_weights.append(weights)
        client_data_counts.append(data_count)

        # Aggregate weights using weighted averaging
        total_data_points = sum(client_data_counts)
        weighted_weights = np.average(model_weights, axis=0, weights=client_data_counts)

        print(f"Aggregated weighted weights: {weighted_weights}")

        return jsonify({"status": "Model update received!", "aggregated_weights": weighted_weights.tolist()})
    except Exception as e:
        return jsonify({"status": "Error processing weights", "error": str(e)}), 400

@app.route("/get_aggregated_model", methods=["GET"])
def get_aggregated_model():
    if not model_weights:
        return jsonify({"status": "No model weights received yet."}), 400

    # Aggregate weights and send back
    total_data_points = sum(client_data_counts)
    weighted_weights = np.average(model_weights, axis=0, weights=client_data_counts)

    return jsonify({"status": "Aggregated model sent", "aggregated_weights": weighted_weights.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
