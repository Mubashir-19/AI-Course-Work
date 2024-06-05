import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create a Sequential model
model = models.Sequential([

    # First convolutional layer: 32 filters, 3x3 size, ReLU activation function
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # First max pooling layer: 2x2 size
    layers.MaxPooling2D((2, 2)),

    # Second convolutional layer: 64 filters, 3x3 size, ReLU activation function
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Second max pooling layer: 2x2 size
    layers.MaxPooling2D((2, 2)),

    # Third convolutional layer: 64 filters, 3x3 size, ReLU activation function
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten the feature maps into a single 1D vector
    layers.Flatten(),

    # Fully connected (dense) layer with 64 units and ReLU activation function
    layers.Dense(64, activation='relu'),

    # Output layer with 10 units (one for each digit) and softmax activation function
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

model.save('digitRecognizer.keras')
