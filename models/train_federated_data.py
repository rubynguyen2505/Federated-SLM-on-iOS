import tensorflow as tf
import tensorflow_federated as tff
from load_federated_data import load_federated_data  # Import the function to load federated data

# Create a simple Keras model
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=128),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Convert Keras model to a TFF model
def create_federated_model():
    keras_model = create_keras_model()
    model = tff.learning.models.from_keras_model(
        keras_model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=[
            tf.TensorSpec([None, 100], tf.int32),  # Input spec for the features
            tf.TensorSpec([None], tf.int32)  # Input spec for the labels (target)
        ])
    return model

# Load the federated data
federated_data = load_federated_data()

# Create the federated learning process
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=create_federated_model,
    client_optimizer_fn=tff.learning.optimizers.build_adam(0.1),
)
state = iterative_process.initialize()

NUM_ROUNDS = 10

# Convert federated_data into a list of datasets (one per client)
federated_data_list = [federated_data.create_tf_dataset_for_client(client) for client in federated_data.client_ids]

# Federated training loop
for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_data_list)
    print(f'Round {round_num}, Metrics: {metrics}')

# Get the trained Keras model from the latest federated state
trained_keras_model = create_keras_model()

# Extract trained weights from the federated state
federated_weights = iterative_process.get_model_weights(state)

# Ensure the weights are correctly formatted before assigning
trained_keras_model.set_weights(federated_weights.trainable)  # Extract only trainable weights

# Convert the trained Keras model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(trained_keras_model)

# Enable Select TF Ops to support dynamic operations
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS  # Allow TF ops for unsupported layers like LSTM
]

# Disable experimental lowering of tensor list ops to avoid the issue
converter._experimental_lower_tensor_list_ops = False

# Optimize the model for mobile devices
converter.optimizations = [tf.lite.Optimize.DEFAULT]  

# Convert the model
tflite_model = converter.convert()

# Save the model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
