import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json

DATASET_NAME = "imdb_reviews"

def load_dataset():
    dataset = tfds.load(DATASET_NAME, split='train', as_supervised=True)
    return dataset

def preprocess_dataset(dataset):
    text_data = [text.numpy().decode("utf-8") for text, _ in dataset]
    labels = [label.numpy() for _, label in dataset]  # Extract the labels as well

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(text_data)

    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)

    return np.array(padded_sequences), np.array(labels), tokenizer  # Return labels as well

if __name__ == "__main__":
    dataset = load_dataset()
    sequences, labels, tokenizer = preprocess_dataset(dataset)
    
    # Save sequences and labels
    np.savez("preprocessed_data.npz", sequences=sequences, labels=labels)  # Save labels here
    
    # Save the tokenizer as a separate file
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        json.dump(tokenizer_json, f)

    print(f"Dataset '{DATASET_NAME}' preprocessed and saved!")

