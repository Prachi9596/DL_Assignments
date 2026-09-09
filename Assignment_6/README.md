# Assignment 3 – Tomato Leaf Disease Classification using CNN

## Aim

Design and implement a Convolutional Neural Network (CNN) for image classification of tomato leaf diseases using TensorFlow/Keras.

## Objective

The objective of this assignment is to understand how CNNs can be used for image classification.

The following tasks are performed:

- Load a real-world image dataset
- Identify the different disease classes
- Split the dataset into training and validation sets
- Organize images into class-wise directories
- Resize images to a fixed size
- Normalize image pixel values
- Design a CNN architecture
- Train the CNN model
- Evaluate the model using validation accuracy and loss
- Perform prediction on a validation image

## Dataset

The **Tomato Leaf Disease** dataset from the `Project-AgML` collection on Hugging Face is used.

The dataset contains images of tomato leaves belonging to different disease/health categories.

The dataset is loaded using:

```python
load_dataset("Project-AgML/tomato_leaf_disease")