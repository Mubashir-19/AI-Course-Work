# Project: MNIST Digit Recognition Using Convolutional Neural Networks

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction
Digit recognition is a fundamental problem in computer vision and machine learning. The MNIST dataset, a collection of handwritten digits, is a standard benchmark for evaluating algorithms. In this project, we build a convolutional neural network (CNN) to recognize digits from the MNIST dataset with high accuracy.

## Dataset
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), with each image being 28x28 pixels. The dataset is divided into:
- Training set: 60,000 images
- Test set: 10,000 images

## Model Architecture
The CNN model architecture consists of the following layers:
1. **Input Layer**: 28x28 grayscale image
2. **Convolutional Layer 1**: 32 filters, 3x3 kernel, ReLU activation
3. **Max Pooling Layer 1**: 2x2 pool size
4. **Convolutional Layer 2**: 64 filters, 3x3 kernel, ReLU activation
5. **Max Pooling Layer 2**: 2x2 pool size
6. **Flatten Layer**: Flattening the 2D arrays into a 1D vector
7. **Dense Layer 1**: 128 units, ReLU activation
8. **Dropout Layer**: 0.5 dropout rate
9. **Dense Layer 2 (Output Layer)**: 10 units (one for each digit), Softmax activation

## Training
The model is trained using the following configurations:
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Batch Size**: 128
- **Epochs**: 3
- **Validation Split**: 20%

## Results
- **Training Accuracy**: 0.99
- **Test Accuracy**: 0.99%

## Conclusion
The convolutional neural network achieved a high accuracy on the MNIST digit recognition task, demonstrating the effectiveness of CNNs in image classification problems. Future work could include exploring deeper architectures or applying the model to more complex datasets.

## References
1. ([AN ENSEMBLE OF SIMPLE CONVOLUTIONAL NEURAL
 NETWORK MODELS FOR MNIST DIGIT RECOGNITION](https://github.com/Mubashir-19/AI-Course-Work/blob/main/MNIST%20Digit%20Recognition%20Paper%202.pdf))
2. ([Assessing Four Neural Networks on Handwritten
 Digit Recognition Dataset (MNIST)](https://github.com/Mubashir-19/AI-Course-Work/blob/main/MNIST%20DIGIT%20Recognition%20Paper%201.pdf)) 
3. [MNIST Database](http://yann.lecun.com/exdb/mnist/)
4. [TensorFlow Documentation](https://www.tensorflow.org/)
5. [Keras Documentation](https://keras.io/)

