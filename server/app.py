from flask import Flask, request, send_file, make_response, Response
import os
import zipfile
from datetime import datetime
import coremltools as ct
import numpy as np
import struct
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random
import gzip
import json
from io import BytesIO
import glob
import shutil
import csv

# Constants
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBED_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10

# Load IMDb
(train_data, val_data, test_data), info = tfds.load(
    'imdb_reviews',
    split=['train[:80%]', 'train[80%:]', 'test'],
    as_supervised=True,
    with_info=True
)

# Prepare raw texts
train_texts = [x.numpy().decode("utf-8") for x, _ in train_data]
train_labels = [int(y.numpy()) for _, y in train_data]
val_texts = [x.numpy().decode("utf-8") for x, _ in val_data]
val_labels = [int(y.numpy()) for _, y in val_data]
test_texts = [x.numpy().decode("utf-8") for x, _ in test_data]
test_labels = [int(y.numpy()) for _, y in test_data]

# Get random subset of test data
subset_size = 2500
indices = random.sample(range(len(test_texts)), subset_size)
test_texts = [test_texts[i] for i in indices]
test_labels = [test_labels[i] for i in indices]

# Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=MAX_LEN)
val_sequences = tokenizer.texts_to_sequences(val_texts)
val_padded = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=MAX_LEN)
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=MAX_LEN)

# Convert to NumPy arrays
train_inputs = np.array(train_padded)
train_labels = np.array(train_labels)
val_inputs = np.array(val_padded)
val_labels = np.array(val_labels)
test_inputs = np.array(test_padded)
test_labels = np.array(test_labels)

def create_pretrained_model():
    # Updatable-friendly model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN, name="embedding"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(64, activation='relu', name="dense1"),
        tf.keras.layers.Dense(2, activation='softmax', name="output")
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Pretraining
    model.fit(train_inputs, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_inputs, val_labels))

    # Save the model
    model.save("imdb_model.keras")
    print("✅ Pretrained model saved as imdb_model.keras")

    # Save tokenizer
    with open("tokenizer.json", "w") as f:
        json.dump({"word_index": tokenizer.word_index}, f)

    print("✅ Tokenizer saved.")


def convert_to_coreml():
    # Load the model
    model = tf.keras.models.load_model("imdb_model.keras")

    model.build(input_shape=(None, MAX_LEN))
    _ = model.get_weights()

    # # Set input type
    input_type = ct.TensorType(shape=(1, MAX_LEN), dtype=np.int32)

    # Convert to CoreML Neural Network
    mlmodel = ct.convert(
        model,
        convert_to="neuralnetwork",
        inputs=[input_type],
        compute_units=ct.ComputeUnit.ALL
    )

    # Enable updatable model
    old_spec = mlmodel.get_spec()
    old_spec.description.metadata.shortDescription = "Updatable IMDB sentiment classifier"
    old_spec.description.metadata.author = "Your Name"
    old_spec.isUpdatable = True

    old_nn = old_spec.neuralNetwork
    # Delete the last layer (softmaxND) as it is incompatible with updatable models
    del old_nn.layers[-1] 
    second_to_last_index = len(old_nn.layers) - 1

    old_spec.neuralNetwork.layers[second_to_last_index].output[0] = "output_r"

    ct.utils.save_spec(old_spec, "imdb_model.mlmodel")
    spec = ct.utils.load_spec("imdb_model.mlmodel")

    # Add the last layer again as a softmax layer (not softmaxND) 
    softmax_layer = spec.neuralNetwork.layers.add()
    softmax_layer.name = "softmax"
    softmax_layer.softmax.MergeFromString(b"")
    softmax_layer.input.append("output_r")
    softmax_layer.output.append("Identity")

    ct.utils.save_spec(spec, "imdb_model.mlmodel")

    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
    # Mark updatable layers
    builder.make_updatable([
        "sequential/output/BiasAdd",
        "sequential/dense1/BiasAdd"
    ])

    # Set up the model for training
    builder.set_categorical_cross_entropy_loss(name="lossLayer", input='Identity')
    builder.set_adam_optimizer(ct.models.neural_network.AdamParams(lr=0.01, batch=32))
    builder.set_epochs(10)

    model_spec = builder.spec
    model_spec.description.metadata.shortDescription = "Updatable IMDB sentiment classifier"
    model_spec.description.metadata.author = "Your Name"
    model_spec.isUpdatable = True

    mlmodel = ct.models.MLModel(model_spec)

    mlmodel.save("imdb_updatable_model.mlpackage")

def get_centralized_keras_model_score():
    model = tf.keras.models.load_model("imdb_model.keras")

    # Fine tuning
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(val_inputs, val_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

    # Predict
    pred_probs = model.predict(test_inputs)
    pred_labels = np.argmax(pred_probs, axis=1)

    # Accuracy and other metrics
    acc = accuracy_score(test_labels, pred_labels)
    precision = precision_score(test_labels, pred_labels, average='weighted')
    recall = recall_score(test_labels, pred_labels, average='weighted')
    f1 = f1_score(test_labels, pred_labels, average='weighted')

    print(f"Fine-tuned Accuracy: {acc:.4f}")
    print(f"Fine-tuned Precision: {precision:.4f}")
    print(f"Fine-tuned Recall: {recall:.4f}")
    print(f"Fine-tuned F1 Score: {f1:.4f}")

    # CSV Logging
    log_file = "centralized_model_metrics.csv"
    write_header = not os.path.exists(log_file)

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["model_type", "accuracy", "precision", "recall", "f1_score"])
        writer.writerow(["centralized", acc, precision, recall, f1])

    return acc, precision, recall, f1

create_pretrained_model()
convert_to_coreml()

def read_compiled_weights(mlmodelc_path):
    layer_bytes = []
    layer_data = {}
    weights = {}
    filename = os.path.join(mlmodelc_path, 'model.espresso.weights')
    with open(filename, 'rb') as f:
        num_layers = struct.unpack('<i', f.read(4))[0]

        f.read(4)

        # Read the number of layers and their sizes
        while len(layer_bytes) < num_layers:
            layer_num, _, num_bytes, _ = struct.unpack('<iiii', f.read(16))
            layer_bytes.append((layer_num, num_bytes))

        # Read the layer data
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

    # Iterate through each layer's weights and biases
    for key in all_keys:
        w_stack = np.stack([w[key]["weights"] for w in weight_dicts])
        b_stack = np.stack([w[key]["bias"] for w in weight_dicts])

        # Calculate the average weights and biases
        avg[key] = {
            "weights": np.mean(w_stack, axis=0),
            "bias": np.mean(b_stack, axis=0)
        }
    return avg

def set_weights_in_model(base_model_path, avg_weights, output_model_path):
    # Load the base model from .mlpackage
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
    # if same client id, different timestamp, delete old model
    for old_file in glob.glob(os.path.join(UPLOAD_FOLDER, f"{client_id}_*.zip")):
        try:
            os.remove(old_file)
            print(f"❌ Deleted old model: {old_file}")
        except Exception as e:
            print(f"⚠️ Failed to delete {old_file}: {e}")
    zip_filename = f"{client_id}_{timestamp}.zip"
    zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)

    file.save(zip_path)
    print(f"✅ Received model and saved to {zip_path}")

    for old_folder in glob.glob(os.path.join(MODEL_FOLDER, f"{client_id}_*")):
        try:
            shutil.rmtree(old_folder)
            print(f"❌ Deleted old extracted model: {old_folder}")
        except Exception as e:
            print(f"⚠️ Failed to delete {old_folder}: {e}")

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
    base_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server', 'imdb_updatable_model.mlpackage', 'Data', 'com.apple.CoreML', 'model.mlmodel')
    output_model_path = './aggregated_model.mlmodel'

    # Find all extracted model folders
    model_dirs = [os.path.join(MODEL_FOLDER, d) for d in os.listdir(MODEL_FOLDER)
                  if os.path.isdir(os.path.join(MODEL_FOLDER, d))]

    model_paths = []
    for d in model_dirs:
        for fname in os.listdir(d):
            if fname.endswith('.mlmodelc'):
                model_paths.append(os.path.join(d, fname))

    if len(model_paths) < 1:
        return 'Need at least 1 model to aggregate', 400

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

@app.route('/get_train_data', methods=['GET'])
def get_train_data():
    subset_size = 50
    # Get a random subset of training data
    indices = random.sample(range(len(val_texts)), subset_size)
    samples = [{"text": val_texts[i], "label": int(val_labels[i])} for i in indices]
    return {"data": samples}, 200

@app.route('/get_test_data', methods=['GET'])
def get_test_data():
    samples = [{"text": text, "label": int(label)} for text, label in zip(test_texts, test_labels)]
    return {"data": samples}, 200

@app.route('/get_test_data_gzip', methods=['GET'])
def get_test_data_gzip():
    samples = [{"text": text, "label": int(label)} for text, label in zip(test_texts, test_labels)]
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="w") as f:
        f.write(json.dumps({"data": samples}).encode('utf-8'))
    buffer.seek(0)
    return Response(buffer, mimetype='application/gzip', headers={'Content-Encoding': 'gzip'})

@app.route('/report_metrics', methods=['POST'])
def report_metrics():
    data = request.get_json()

    required_fields = [
        'client_id', 'round', 'accuracy', 'precision', 'recall', 'f1_score', 
        'log_loss', 'confusion_matrix', 'evaluation_time_ms', 'prediction_confidence', 
        'per_class_precision', 'per_class_recall'
    ]
    
    if not all(field in data for field in required_fields):
        return 'Missing one or more required fields', 400

    log_file = 'federated_metrics_log.csv'
    file_exists = os.path.isfile(log_file)

    try:
        with open(log_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=required_fields)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'client_id': data['client_id'],
                'round': data['round'],
                'accuracy': data['accuracy'],
                'precision': data['precision'],
                'recall': data['recall'],
                'f1_score': data['f1_score'],
                'log_loss': data['log_loss'],
                'confusion_matrix': data['confusion_matrix'],
                'evaluation_time_ms': data['evaluation_time_ms'],
                'prediction_confidence': data['prediction_confidence'],
                'per_class_precision': data['per_class_precision'],
                'per_class_recall': data['per_class_recall'],
            })

        print(f"✅ Logged metrics for client '{data['client_id']}' at round {data['round']}")
        return 'Metrics logged successfully', 200

    except Exception as e:
        print(f"❌ Failed to log metrics: {e}")
        return f"Logging failed: {e}", 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
