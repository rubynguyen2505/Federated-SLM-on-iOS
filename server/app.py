from flask import Flask, request
import numpy as np

app = Flask(__name__)

model_weights = []

@app.route("/upload_model", methods=["POST"])
def upload_model():
    global model_weights
    weights = np.array(request.json["weights"])
    model_weights.append(weights)

    # Aggregate weights (for now, just average them)
    aggregated_weights = np.mean(model_weights, axis=0)
    print(f"Aggregated weights: {aggregated_weights}")

    return {"status": "Model update received!"}

if __name__ == "__main__":
    app.run(port=5000)
