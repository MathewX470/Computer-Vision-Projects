# Image Classification with CNN

A Convolutional Neural Network (CNN) built with TensorFlow/Keras for image classification using the CIFAR-10 dataset. The project includes two implementations: a baseline CNN model and an enhanced version with data augmentation for improved accuracy.

## 📋 Overview

This project implements deep learning CNN models to classify images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Two trained models are available - one with standard training and another with data augmentation techniques for better generalization.

## 🎯 Features

- **Multi-class Classification**: Classifies images into 10 distinct categories
- **Two Model Versions**: Standard CNN and data-augmented CNN for comparison
- **Data Augmentation**: Includes random flip, rotation, and zoom augmentation
- **Pre-trained Models**: Both models saved and ready for inference
- **Visualization**: Training accuracy and loss plots for performance analysis
- **Complete Training Pipeline**: From data loading to model evaluation

## 🏗️ Model Architecture

### Baseline CNN Model
```
Input Layer: 32x32x3 (RGB images)
├── Conv2D: 32 filters, 3x3 kernel, ReLU activation
├── MaxPooling2D: 2x2 pool size
├── Conv2D: 64 filters, 3x3 kernel, ReLU activation
├── MaxPooling2D: 2x2 pool size
├── Flatten
├── Dense: 128 units, ReLU activation
├── Dropout: 0.5 rate
└── Dense: 10 units, Softmax activation (output)
```

### Data Augmentation Model
```
Input Layer: 32x32x3 (RGB images)
├── Data Augmentation Layer (Random Flip, Rotation, Zoom)
├── Conv2D: 32 filters, 3x3 kernel, ReLU activation
├── MaxPooling2D: 2x2 pool size
├── Conv2D: 64 filters, 3x3 kernel, ReLU activation
├── MaxPooling2D: 2x2 pool size
├── Conv2D: 64 filters, 3x3 kernel, ReLU activation
├── MaxPooling2D: 2x2 pool size
├── Flatten
├── Dense: 128 units, ReLU activation
├── Dropout: 0.5 rate
└── Dense: 10 units, Softmax activation (output)
```

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **OpenCV**: Image processing (for testing)

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install tensorflow numpy matplotlib
```

### Training the Baseline Model

1. Open [image_classification_with_CNN.ipynb](image_classification_with_CNN.ipynb) in Jupyter Notebook or VS Code
2. Run all cells sequentially to:
   - Load and preprocess the CIFAR-10 dataset
   - Build the CNN model
   - Train the model (10 epochs)
   - Evaluate performance
   - Save the trained model

### Training the Data Augmentation Model

1. Open [data_augmentation_image_classification.ipynb](data_augmentation_image_classification.ipynb)
2. Run all cells to:
   - Load and preprocess the CIFAR-10 dataset
   - Apply data augmentation (random flip, rotation, zoom)
   - Build and train the enhanced CNN model
   - Compare performance with baseline
   - Save the augmented model

## 📊 Model Performance

- **Training**: 10 epochs with batch size 64
- **Validation Split**: 20% of training data
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Regularization**: Dropout (0.5)

## 🎨 Dataset

**CIFAR-10 Dataset**
- 60,000 32x32 color images
- 10 classes with 6,000 images per class
- 50,000 training images
- 10,000 test images

## 📚 References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)

---
⭐ Star this repository if you found it helpful!
