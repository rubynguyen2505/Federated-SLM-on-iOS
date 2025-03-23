import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import pickle

print("TensorFlow version: {}".format(tf.__version__))
print("TensorFlow Federated version: {}".format(tff.__version__))

def load_federated_data():
    # Load the preprocessed data and labels
    data = np.load("preprocessed_data.npz")['sequences']
    labels = np.load("preprocessed_data.npz")['labels']
    
    # Load the tokenizer for consistent text handling
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Create client datasets (you can split the data as needed)
    client_data = []
    num_clients = 3
    client_size = len(data) // num_clients
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = (i + 1) * client_size
        client_data.append((data[start_idx:end_idx], labels[start_idx:end_idx]))  # Save both input and label

    # Define dataset function to return federated data
    def dataset_fn(client_id):
        # Get the client index from client_id string (e.g., "client_0" -> 0)
        client_index = int(client_id.split('_')[1])
        client_inputs, client_labels = client_data[client_index]  # Access both inputs and labels
        
        # Ensure labels are cast to int32
        client_labels = client_labels.astype(np.int32)

        client_dataset = tf.data.Dataset.from_tensor_slices((client_inputs, client_labels))  # Provide both
        client_dataset = client_dataset.batch(32)  # Adjust batch size as necessary
        return client_dataset

    # Use TFF's ClientData interface, passing string client IDs
    federated_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        [f"client_{i}" for i in range(num_clients)],  # Client IDs as strings
        dataset_fn
    )

    return federated_data


if __name__ == "__main__":
    federated_data = load_federated_data()
    print(f"Federated data: {federated_data}")
    print(f"Client IDs: {federated_data.client_ids}")