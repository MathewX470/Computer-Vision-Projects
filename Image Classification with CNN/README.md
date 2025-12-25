# Image Classification with CNN

A Convolutional Neural Network (CNN) built with TensorFlow/Keras for image classification using the CIFAR-10 dataset. The model achieves accurate classification across 10 different object categories.

## 📋 Overview

This project implements a deep learning CNN model to classify images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The model is trained on the CIFAR-10 dataset and can predict the class of new images with confidence scores.

## 🎯 Features

- **Multi-class Classification**: Classifies images into 10 distinct categories
- **CNN Architecture**: Uses convolutional layers, max pooling, and dropout for robust learning
- **Pre-trained Model**: Includes a saved model ready for inference
- **Visualization**: Training accuracy and loss plots
- **Test Suite**: Separate notebook for testing the model on custom images

## 🏗️ Model Architecture

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

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **OpenCV**: Image processing (for testing)

## 📁 Project Structure

```
Image Classification with CNN/
├── image_classification_with_CNN.ipynb  # Main training notebook
├── test_model.ipynb                     # Model testing notebook
├── cnn_image_classifier.keras           # Saved trained model
├── test_images/                         # Sample images for testing
│   ├── airplane.jpg
│   ├── automobile.jpg
│   ├── frog.jpg
│   ├── horse.jpg
│   ├── ship.jpg
│   └── truck.jpg
└── README.md
```

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install tensorflow numpy matplotlib opencv-python
```

### Training the Model

1. Open `image_classification_with_CNN.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially to:
   - Load and preprocess the CIFAR-10 dataset
   - Build the CNN model
   - Train the model (10 epochs)
   - Evaluate performance
   - Save the trained model

### Testing the Model

1. Open `test_model.ipynb`
2. Load the pre-trained model
3. Place test images in the `test_images/` folder
4. Run the notebook to see predictions with confidence scores

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

### Classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## 💡 Usage Example

```python
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load model
model = load_model("cnn_image_classifier.keras", compile=False)

# Prepare image
img = cv2.imread("your_image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (32, 32))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
```

## 📈 Future Improvements

- [ ] Implement data augmentation for better generalization
- [ ] Experiment with deeper architectures (ResNet, VGG)
- [ ] Add learning rate scheduling
- [ ] Implement early stopping
- [ ] Deploy as web application
- [ ] Add confusion matrix visualization
- [ ] Support for custom datasets

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👤 Author

**MathewX470**
- GitHub: [@MathewX470](https://github.com/MathewX470)

## 📚 References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)

---
⭐ Star this repository if you found it helpful!
