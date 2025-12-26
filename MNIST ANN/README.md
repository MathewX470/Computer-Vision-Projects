# MNIST Digit Classification with Artificial Neural Networks

A deep learning project that implements an Artificial Neural Network (ANN) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

## 📋 Overview

This project demonstrates the application of neural networks for image classification tasks. It explores different ANN architectures, comparing a simple single-layer model with more complex hidden-layer configurations to achieve optimal digit recognition accuracy.

## 🎯 Features

- **MNIST Dataset Integration**: Loads and preprocesses the classic MNIST handwritten digit dataset (70,000 grayscale images)
- **Data Visualization**: Displays sample images with their corresponding labels
- **Image Normalization**: Scales pixel values from 0-255 to 0-1 for better model performance
- **Multiple Model Architectures**:
  - Simple single-layer model (baseline)
  - Hidden layer model with 100 neurons (improved performance)
  - Flatten layer implementation (optimized preprocessing)
- **Performance Evaluation**: Comprehensive accuracy metrics and confusion matrix visualization
- **Confusion Matrix Heatmaps**: Visual representation of model predictions vs. actual labels

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow 2.x**: Deep learning framework
- **Keras**: High-level neural networks API
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization


## 🚀 Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install tensorflow numpy matplotlib seaborn
```

### Running the Project

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mnist-ann.git
cd mnist-ann
```

2. Open the Jupyter notebook:
```bash
jupyter notebook mnist_ann.ipynb
```

3. Run all cells sequentially to:
   - Load the MNIST dataset
   - Visualize sample images
   - Train different model architectures
   - Evaluate model performance

## 📊 Model Architectures

### Model 1: Simple Single-Layer Network
```python
- Input Layer: 784 neurons (28×28 flattened)
- Output Layer: 10 neurons (sigmoid activation)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
```

### Model 2: Hidden Layer Network
```python
- Input Layer: 784 neurons (28×28 flattened)
- Hidden Layer: 100 neurons (ReLU activation)
- Output Layer: 10 neurons (sigmoid activation)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
```

### Model 3: Optimized with Flatten Layer
```python
- Flatten Layer: Converts 28×28 images to 784-dimensional vectors
- Hidden Layer: 100 neurons (ReLU activation)
- Output Layer: 10 neurons (sigmoid activation)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
```

## 📈 Results

The project demonstrates progressive improvement in accuracy:

1. **Simple Model**: Baseline accuracy with single-layer architecture
2. **Hidden Layer Model**: Significant accuracy improvement with deeper network
3. **Optimized Model**: Streamlined preprocessing using Flatten layer while maintaining high accuracy

Each model's performance is evaluated using:
- Test accuracy metrics
- Confusion matrices showing prediction patterns
- Visual heatmaps for easy interpretation

## 🔍 Key Insights

- **Normalization**: Scaling pixel values (0-1 range) significantly improves training stability
- **Hidden Layers**: Adding hidden layers with ReLU activation dramatically improves classification accuracy
- **Flatten Layer**: Using Keras's Flatten layer eliminates manual reshaping and simplifies the pipeline
- **Architecture Depth**: Deeper networks capture more complex patterns in handwritten digits

## 📝 Implementation Details

### Data Preprocessing
- Images normalized by dividing by 255
- Original shape: 28×28 pixels
- Flattened to 784-dimensional vectors for dense layer input

### Training Configuration
- **Epochs**: 5
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Sparse Categorical Crossentropy (suitable for multi-class classification)
- **Metrics**: Accuracy

## 🎓 Learning Outcomes

This project demonstrates:
- Building neural networks with TensorFlow/Keras
- Data preprocessing for computer vision tasks
- Comparing different neural network architectures
- Model evaluation using confusion matrices
- Visualizing deep learning results

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun and collaborators
- TensorFlow and Keras teams for excellent documentation
- The deep learning community for inspiration and best practices

---

⭐ Star this repository if you found it helpful!
