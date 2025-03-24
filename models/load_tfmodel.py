import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
print(input_details)
