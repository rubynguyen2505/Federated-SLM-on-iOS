from flask import Flask, request, send_file, make_response
import os
import zipfile
from datetime import datetime
import coremltools as ct
import numpy as np
import struct

def read_compiled_weights(mlmodelc_path):
    layer_bytes = []
    layer_data = {}
    weights = {}
    filename = os.path.join(mlmodelc_path, 'model.espresso.weights')
    with open(filename, 'rb') as f:
        num_layers = struct.unpack('<i', f.read(4))[0]

        f.read(4)

        while len(layer_bytes) < num_layers:
            layer_num, _, num_bytes, _ = struct.unpack('<iiii', f.read(16))
            layer_bytes.append((layer_num, num_bytes))

        for layer_num, num_bytes in layer_bytes:
            data = struct.unpack('f' * (num_bytes // 4), f.read(num_bytes))
            layer_data[layer_num] = data
        
        weights['sequential/dense1/BiasAdd'] = {
            "weights": np.array(layer_data[5]),
            "bias": np.array(layer_data[3])
        }
        weights['sequential/output/BiasAdd'] = {
            "weights": np.array(layer_data[9]),
            "bias": np.array(layer_data[7])
        }
        return weights

def fedavg(weight_dicts):
    avg = {}
    n = len(weight_dicts)
    all_keys = weight_dicts[0].keys()

    for key in all_keys:
        w_stack = np.stack([w[key]["weights"] for w in weight_dicts])
        b_stack = np.stack([w[key]["bias"] for w in weight_dicts])

        avg[key] = {
            "weights": np.mean(w_stack, axis=0),
            "bias": np.mean(b_stack, axis=0)
        }
    return avg

def set_weights_in_model(base_model_path, avg_weights, output_model_path):
    # Load model from .mlpackage
    model = ct.models.MLModel(base_model_path)
    spec = model.get_spec()

    # Update weights in the model specification
    for layer in spec.neuralNetwork.layers:
        if layer.name in avg_weights:
            print(np.array(layer.innerProduct.weights.floatValue)[0])
            layer.innerProduct.weights.floatValue[:] = avg_weights[layer.name]["weights"].flatten().tolist()
            layer.innerProduct.bias.floatValue[:] = avg_weights[layer.name]["bias"].flatten().tolist()

    # Save the updated model
    ct.models.MLModel(spec).save(output_model_path)
    print(f"✅ Aggregated model saved to: {output_model_path}")

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
MODEL_FOLDER = './models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return 'No model file part', 400

    file = request.files['model']
    if file.filename == '':
        return 'No selected file', 400

    client_id = request.remote_addr.replace('.', '_') 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f"{client_id}_{timestamp}.zip"
    zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)
    file.save(zip_path)
    print(f"✅ Received model and saved to {zip_path}")

    extract_dir = os.path.join(MODEL_FOLDER, f"{client_id}_{timestamp}")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Unzipped to {extract_dir}")
        return 'Model uploaded and unzipped successfully', 200
    except Exception as e:
        print(f"❌ Failed to unzip: {e}")
        return f"Unzipping failed: {str(e)}", 500

@app.route('/aggregate', methods=['POST'])
def aggregate_models():
    base_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'imdb_updatable_model.mlpackage', 'Data', 'com.apple.CoreML', 'model.mlmodel')
    output_model_path = './aggregated_model.mlmodel'

    # Find all extracted model folders
    model_dirs = [os.path.join(MODEL_FOLDER, d) for d in os.listdir(MODEL_FOLDER)
                  if os.path.isdir(os.path.join(MODEL_FOLDER, d))]

    model_paths = []
    for d in model_dirs:
        for fname in os.listdir(d):
            if fname.endswith('.mlmodelc'):
                model_paths.append(os.path.join(d, fname))

    if len(model_paths) < 2:
        return 'Need at least 2 models to aggregate', 400

    # Extract weights from each model
    all_weights = [read_compiled_weights(model_path) for model_path in model_paths]
    avg_weights = fedavg(all_weights)

    # Set weights in the base model and save the aggregated model
    set_weights_in_model(base_model_path, avg_weights, output_model_path)

    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')  # Use timestamp as the version
    model_version_file = './model_version.txt'
    
    with open(model_version_file, 'w') as f:
        f.write(model_version)
    
    print(f"✅ Aggregated model version: {model_version}")

    return f"✅ Aggregated {len(model_paths)} models into {output_model_path}", 200

@app.route('/download', methods=['GET'])
def download_aggregated_model():
    output_model_path = './aggregated_model.mlmodel'
    if not os.path.exists(output_model_path):
        return 'Aggregated model not found', 404
    
    model_version_file = './model_version.txt'
    if os.path.exists(model_version_file):
        with open(model_version_file, 'r') as f:
            model_version = f.read().strip()
    else:
        model_version = 'v1.0.0'

    print(f"✅ Model Version: {model_version}")
    
    response = make_response(send_file(output_model_path, as_attachment=True))
    response.headers['Model-Version'] = model_version
    return response

@app.route('/metrics', methods=['POST'])
def receive_metrics():
    data = request.get_json()
    
    accuracy = data.get('accuracy', None)
    loss = data.get('loss', None)
    model_version = data.get('model_version', None)
    
    if not accuracy or not loss or not model_version:
        return 'Missing required data (accuracy, loss, or model_version)', 400
    
    print(f"Received metrics - Model Version: {model_version}, Accuracy: {accuracy}, Loss: {loss}")
    
    metrics_log = './metrics_log.txt'
    with open(metrics_log, 'a') as f:
        f.write(f"{datetime.now()} - Model Version: {model_version}, Accuracy: {accuracy}, Loss: {loss}\n")
    
    return 'Metrics received successfully', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
