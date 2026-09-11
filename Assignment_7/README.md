# Transfer Learning for Image Classification

## Assignment 7

**Implement transfer learning using pre-trained AlexNet, VGG16, ResNet50, and EfficientNetB0 models for image classification, and compare their performance.**

---

## Project Overview

This project demonstrates image classification using Convolutional Neural Networks (CNNs) and transfer learning.

Four popular CNN architectures are implemented and their performance is compared:

* AlexNet
* VGG16
* ResNet50
* EfficientNetB0

The **CIFAR-10 dataset** is used for image classification. The dataset contains 10 different image classes such as airplanes, automobiles, birds, cats, dogs, and trucks.

The pretrained VGG16, ResNet50, and EfficientNetB0 models use **ImageNet weights**. The pretrained convolutional layers are frozen, and new classification layers are added for the 10 CIFAR-10 classes.

> **Note:** TensorFlow/Keras does not provide an official ImageNet-pretrained AlexNet model through `tensorflow.keras.applications`. Therefore, the AlexNet implementation in this project uses the standard AlexNet architecture and is trained on CIFAR-10.

---

## Objectives

The main objectives of this assignment are:

1. Understand transfer learning.
2. Load and preprocess the CIFAR-10 dataset.
3. Use pretrained CNN architectures for image classification.
4. Freeze the pretrained convolutional layers.
5. Add custom classification layers.
6. Train the models on CIFAR-10.
7. Evaluate the models using test accuracy and test loss.
8. Compare the performance of AlexNet, VGG16, ResNet50, and EfficientNetB0.
9. Visualize model performance using graphs.
10. Perform predictions on unseen test images.

---

## Dataset

### CIFAR-10

CIFAR-10 is a popular image classification dataset containing:

* **60,000 color images**
* **50,000 training images**
* **10,000 testing images**
* Image size: **32 × 32 pixels**
* Number of classes: **10**
* Image type: RGB

### Classes

| Label | Class      |
| ----: | ---------- |
|     0 | Airplane   |
|     1 | Automobile |
|     2 | Bird       |
|     3 | Cat        |
|     4 | Deer       |
|     5 | Dog        |
|     6 | Frog       |
|     7 | Horse      |
|     8 | Ship       |
|     9 | Truck      |

The dataset can be loaded directly using TensorFlow/Keras:

```python
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

---

## Models Used

### 1. AlexNet

AlexNet is a deep convolutional neural network originally developed for large-scale image classification.

The architecture contains:

* Convolutional layers
* Max pooling layers
* Fully connected layers
* ReLU activation
* Dropout
* Softmax output layer

The implementation in this project is adapted for the 10 CIFAR-10 classes.

---

### 2. VGG16

VGG16 is a convolutional neural network consisting of 16 layers with learnable weights.

In this project:

```python
VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

is used.

The original ImageNet classification layer is removed using:

```python
include_top=False
```

The pretrained layers are frozen using:

```python
base_model.trainable = False
```

A new classifier is added for CIFAR-10.

---

### 3. ResNet50

ResNet50 is a 50-layer deep CNN that uses residual connections.

The model is loaded using:

```python
ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

The pretrained layers are frozen and a new classifier is added.

ResNet helps solve the degradation problem that can occur when training very deep networks.

---

### 4. EfficientNetB0

EfficientNetB0 is a lightweight and computationally efficient CNN architecture.

It is loaded with ImageNet pretrained weights:

```python
EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

The pretrained feature extraction layers are frozen and a new classifier is trained for CIFAR-10.

---

##  Transfer Learning Process

The general transfer learning process used in this project is:

```text
CIFAR-10 Image
      ↓
Resize to 224 × 224
      ↓
Model-specific preprocessing
      ↓
Pretrained CNN
      ↓
Feature Extraction
      ↓
Global Average Pooling
      ↓
Dense Layer
      ↓
Dropout
      ↓
10-Class Softmax
      ↓
Predicted Class
```

---

##  Preprocessing

The original CIFAR-10 images have a size of:

```text
32 × 32 × 3
```

The pretrained models use:

```text
224 × 224 × 3
```

Therefore, the images are resized:

```python
image = tf.image.resize(
    image,
    (224, 224)
)
```

Each model uses its corresponding preprocessing function.

### VGG16

```python
vgg_preprocess
```

### ResNet50

```python
resnet_preprocess
```

### EfficientNetB0

```python
efficientnet_preprocess
```

For AlexNet, pixel values are normalized:

```python
image = image / 255.0
```

---

## One-Hot Encoding

The CIFAR-10 labels are converted into one-hot encoded vectors.

```python
y_train_cat = to_categorical(
    y_train,
    NUM_CLASSES
)

y_test_cat = to_categorical(
    y_test,
    NUM_CLASSES
)
```

For example, the label:

```text
3
```

representing `cat` becomes:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

Each label contains 10 values because CIFAR-10 has 10 classes.

---

## Custom Classification Layers

For VGG16, ResNet50, and EfficientNetB0, the original ImageNet classification layer is removed.

The following layers are added:

```python
layers.GlobalAveragePooling2D()

layers.Dense(
    256,
    activation='relu'
)

layers.Dropout(0.5)

layers.Dense(
    NUM_CLASSES,
    activation='softmax'
)
```

### GlobalAveragePooling2D

This is a pooling layer that reduces the spatial dimensions of the feature maps.

### Dense Layer

The Dense layer contains 256 neurons and uses the ReLU activation function.

### Dropout

A dropout rate of 0.5 is used to reduce overfitting.

### Output Layer

The final Dense layer contains 10 neurons because CIFAR-10 has 10 classes.

The softmax activation produces class probabilities.

---

##  Freezing Pretrained Layers

For VGG16, ResNet50, and EfficientNetB0:

```python
base_model.trainable = False
```

is used.

This prevents the pretrained weights from being changed during initial training.

The pretrained model acts as a feature extractor, while the newly added classification layers learn to classify CIFAR-10 images.

---

##  Training Configuration

The following configuration is used:

| Parameter         | Value                    |
| ----------------- | ------------------------ |
| Dataset           | CIFAR-10                 |
| Image Size        | 224 × 224                |
| Batch Size        | 32                       |
| Number of Classes | 10                       |
| Optimizer         | Adam                     |
| Loss Function     | Categorical Crossentropy |
| Activation        | ReLU                     |
| Output Activation | Softmax                  |
| Dropout           | 0.5                      |
| Epochs            | 5                        |

---

## Required Libraries

Install the required packages using:

```bash
pip install tensorflow numpy pandas matplotlib
```

If using Jupyter Notebook:

```bash
pip install notebook
```

---

## ▶️ How to Run

### Step 1: Clone the repository

```bash
git clone <your-repository-url>
```

### Step 2: Open the project

Open the project folder in VS Code or Jupyter Notebook.

### Step 3: Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib
```

### Step 4: Open the notebook

Open the `.ipynb` file containing the assignment code.

### Step 5: Run the cells sequentially

Run the cells in order:

```text
1. Import libraries
2. Check GPU
3. Load CIFAR-10
4. Define class names
5. Display sample images
6. One-hot encoding
7. Define image size and batch size
8. Create dataset
9. Create VGG16
10. Create ResNet50
11. Create EfficientNetB0
12. Create AlexNet
13. Create all models
14. Display model summaries
15. Create datasets
16. Train models
17. Evaluate models
18. Compare performance
19. Plot graphs
20. Perform predictions
```

---

## Performance Evaluation

The models are evaluated using:

### Test Accuracy

Accuracy represents the percentage of test images correctly classified.

```python
model.evaluate(test_dataset)
```

Higher accuracy indicates better classification performance.

### Test Loss

Loss represents the difference between the predicted probability distribution and the actual labels.

Lower loss generally indicates better performance.

---

## Model Comparison

The final comparison is stored in a Pandas DataFrame:

```python
results = pd.DataFrame({

    'Model': [
        'AlexNet',
        'VGG16',
        'ResNet50',
        'EfficientNetB0'
    ],

    'Test Accuracy': [
        alexnet_accuracy,
        vgg_accuracy,
        resnet_accuracy,
        efficientnet_accuracy
    ],

    'Test Loss': [
        alexnet_loss,
        vgg_loss,
        resnet_loss,
        efficientnet_loss
    ]
})
```

The resulting table contains:

| Model          |           Test Accuracy |               Test Loss |
| -------------- | ----------------------: | ----------------------: |
| AlexNet        | Obtained after training | Obtained after training |
| VGG16          | Obtained after training | Obtained after training |
| ResNet50       | Obtained after training | Obtained after training |
| EfficientNetB0 | Obtained after training | Obtained after training |

The actual values depend on the training environment and number of epochs.

---

## Visualizations

The project generates the following graphs:

### 1. Accuracy Comparison

A bar chart compares the test accuracy of all four models.

### 2. Loss Comparison

A bar chart compares the test loss of all four models.

### 3. Validation Accuracy

A line graph compares validation accuracy across epochs.

### 4. Validation Loss

A line graph compares validation loss across epochs.

---

## Image Prediction

A sample image from the CIFAR-10 test dataset is selected and passed through all four models.

Each model predicts one of the ten classes:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

The predicted class is obtained using:

```python
np.argmax(prediction)
```

The predicted numerical class is then converted into its corresponding class name using:

```python
class_names[predicted_class]
```

---

## 🏆 Best Model

The best model is selected based on the highest test accuracy:

```python
best_model = results.loc[
    results['Test Accuracy'].idxmax()
]
```

The model with the highest test accuracy is considered the best-performing model in this experiment.
