import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json
import coremltools as ct
from coremltools.proto import FeatureTypes_pb2 as ft
from coremltools.models import datatypes

# Constants
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBED_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5

# Load IMDb
(train_data, test_data), info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# Prepare raw texts
train_texts = [x.numpy().decode("utf-8") for x, _ in train_data]
train_labels = [int(y.numpy()) for _, y in train_data]

# Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=MAX_LEN)

# Convert to NumPy arrays
train_inputs = np.array(train_padded)
train_labels = np.array(train_labels)

# Updatable-friendly model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN, name="embedding"),
    tf.keras.layers.Flatten(name="flatten"),
    tf.keras.layers.Dense(64, activation='relu', name="dense1"),
    tf.keras.layers.Dense(2, activation='softmax', name="output")
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
model.fit(train_inputs, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)


model.build(input_shape=(None, MAX_LEN))
_ = model.get_weights()

# # Set input type
input_type = ct.TensorType(shape=(1, MAX_LEN), dtype=np.int32)

# Convert to neuralnetwork (REQUIRED for updatable)
mlmodel = ct.convert(
    model,
    convert_to="neuralnetwork",
    inputs=[input_type],
    compute_units=ct.ComputeUnit.ALL
)

# Enable updatable and mark layers
old_spec = mlmodel.get_spec()
old_spec.description.metadata.shortDescription = "Updatable IMDB sentiment classifier"
old_spec.description.metadata.author = "Your Name"
old_spec.isUpdatable = True

old_nn = old_spec.neuralNetwork
del old_nn.layers[-1] 
second_to_last_index = len(old_nn.layers) - 1

old_spec.neuralNetwork.layers[second_to_last_index].output[0] = "output_r"

ct.utils.save_spec(old_spec, "imdb_model.mlmodel")
spec = ct.utils.load_spec("imdb_model.mlmodel")

softmax_layer = spec.neuralNetwork.layers.add()
softmax_layer.name = "softmax"
softmax_layer.softmax.MergeFromString(b"")
softmax_layer.input.append("output_r")
softmax_layer.output.append("Identity")

ct.utils.save_spec(spec, "imdb_model.mlmodel")

builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
builder.make_updatable([
    "sequential/output/BiasAdd",
    "sequential/dense1/BiasAdd",
])

builder.set_categorical_cross_entropy_loss(name="lossLayer", input='Identity')
builder.set_adam_optimizer(ct.models.neural_network.AdamParams(lr=0.01, batch=32))
builder.set_epochs(5)

builder.inspect_layers()
model_spec = builder.spec
print(model_spec.description.trainingInput)
model_spec.description.metadata.shortDescription = "Updatable IMDB sentiment classifier"
model_spec.description.metadata.author = "Your Name"
model_spec.isUpdatable = True

mlmodel = ct.models.MLModel(model_spec)
mlmodel.save("imdb_updatable_model.mlpackage")
# Save tokenizer
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as f:
    json.dump({"word_index": tokenizer.word_index}, f)


print("✅ Tokenizer saved.")
