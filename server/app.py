from flask import Flask, request, jsonify
import numpy as np
import threading

app = Flask(__name__)

# Thread lock for safe updates
lock = threading.Lock()

# Global storage
model_weights = []
client_data_counts = []
expected_clients = 5  # Adjust as needed

# Tracking federated rounds
federated_round = 0
federated_accuracies = []
communication_overheads = []

@app.route("/upload_model", methods=["POST"])
def upload_model():
    global model_weights, client_data_counts, federated_round, federated_accuracies, communication_overheads

    try:
        data = request.json
        weights = np.array(data["weights"])
        data_count = data["data_count"]

        with lock:
            model_weights.append(weights)
            client_data_counts.append(data_count)

            # Aggregate only when all expected clients send updates
            if len(model_weights) >= expected_clients:
                federated_round += 1  # Increment round

                # Perform weighted averaging
                total_data_points = sum(client_data_counts)
                aggregated_weights = np.average(model_weights, axis=0, weights=client_data_counts)

                # Simulate accuracy improvement over rounds
                noise = np.random.normal(0, 0.5)  # Small random variation
                accuracy = 70 + 15 * (1 - np.exp(-0.2 * federated_round)) + noise
                federated_accuracies.append(accuracy)

                # Simulate communication overhead
                communication_cost = np.log1p(federated_round) * 5 + np.random.normal(0, 0.3)
                communication_overheads.append(communication_cost)

                # Reset model weights for next round
                model_weights.clear()
                client_data_counts.clear()

                return jsonify({
                    "status": "Model aggregated!",
                    "aggregated_weights": aggregated_weights.tolist(),
                    "federated_round": federated_round,
                    "accuracy": accuracy,
                    "communication_cost": communication_cost
                })

        return jsonify({"status": "Model update received, waiting for more clients."})
    
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


@app.route("/get_training_metrics", methods=["GET"])
def get_training_metrics():
    with lock:
        if not federated_accuracies:
            return jsonify({"status": "No training metrics available yet."}), 400
        
        return jsonify({
            "rounds": list(range(1, federated_round + 1)),
            "federated_accuracies": federated_accuracies,
            "communication_overheads": communication_overheads
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
