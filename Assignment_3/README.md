# Assignment 3 – Tomato Leaf Disease Classification Using CNN

## Aim

Design and implement a Convolutional Neural Network (CNN) for image classification of tomato leaf diseases using TensorFlow/Keras.

## Objective

The objectives of this assignment are:

- Load a real-world image dataset.
- Explore the dataset and identify its classes.
- Check the dataset for missing values.
- Split the dataset into training and validation sets.
- Organize images into class-wise directories.
- Resize images to a fixed size.
- Normalize image pixel values.
- Design and implement a CNN model.
- Train the CNN using the training dataset.
- Evaluate the model using validation loss and accuracy.
- Predict the disease class of a tomato leaf image.

## Dataset

The **Tomato Leaf Disease** dataset is used for this assignment.

The dataset is loaded using the Hugging Face `datasets` library:

```python
dataset = load_dataset("Project-AgML/tomato_leaf_disease")