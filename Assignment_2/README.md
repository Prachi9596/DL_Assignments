# Assignment 2 – Multilayer Perceptron (MLP) for Classification

## Aim

Design and implement a Multilayer Perceptron (MLP) for classification of the Wine dataset and evaluate its performance using accuracy and a confusion matrix.

## Objective

The objective of this assignment is to understand the implementation of a basic Artificial Neural Network (ANN) / Multilayer Perceptron for a multiclass classification problem.

The assignment covers:

- Loading the Wine dataset
- Exploring the dataset
- Checking for missing values
- Performing data preprocessing
- Normalizing the features
- Splitting the dataset into training and testing sets
- Designing an MLP architecture
- Applying ReLU and Softmax activation functions
- Using Dropout and L2 regularization
- Training the model using the Adam optimizer
- Applying Early Stopping
- Evaluating the model using accuracy
- Generating and visualizing a confusion matrix
- Generating a classification report

## Dataset

The **Wine dataset** provided by Scikit-learn is used for this assignment.

The dataset contains chemical analysis results of wines belonging to three different classes.

### Dataset Details

- Number of samples: 178
- Number of features: 13
- Number of classes: 3
- Target classes: 0, 1, 2
- Task: Multiclass Classification

### Features

The 13 input features are:

1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

## Technologies and Libraries Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## MLP Architecture

The Multilayer Perceptron consists of the following layers:

```text
Input Layer
13 Features
    
Dense Layer
32 Neurons + ReLU
L2 Regularization
     
Dropout
30%
     
Dense Layer
16 Neurons + ReLU
L2 Regularization
     
Dropout
30%
     
Output Layer
3 Neurons + Softmax