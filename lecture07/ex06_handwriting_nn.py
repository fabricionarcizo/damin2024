#
# MIT License
#
# Copyright (c) 2024 Fabricio Batista Narcizo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#/

"""This script demonstrates a simple neural network model for handwritten digit
classification."""

# Import the required libraries.
import tf2onnx

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model


# Load the dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data.
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Convert labels to categorical one-hot encoding.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model.
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(10, activation="softmax")
])

# Save the model structure as an image.
plot_model(
    model, to_file="./lecture07/models/mnist_model.png",
    show_shapes=True, show_layer_names=True
)

# Compile the model.
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# Train the model.
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model.
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}.")

# Convert the model to ONNX format.
model.output_names = [ "output" ]
input_signature = [
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="digit")
]
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

# Save the ONNX model to a file.
with open("./lecture07/models/mnist_model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

# Show some test data and the actual and predicted classes in a 2x3 plot.
NUM_SAMPLES = 6
indices = np.random.choice(len(x_test), NUM_SAMPLES, replace=False)
sample_images = x_test[indices]
sample_labels = y_test[indices]
predictions = model.predict(sample_images)

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
axes = axes.flatten()

for i in range(NUM_SAMPLES):
    axes[i].imshow(sample_images[i], cmap="gray")
    axes[i].set_title(
        f"Actual: {np.argmax(sample_labels[i])}, "
        f"Predicted: {np.argmax(predictions[i])}"
    )
    axes[i].axis("off")

plt.tight_layout()
plt.show()
