# Telco Customer Churn Prediction using Artificial Neural Networks

A deep learning project that predicts customer churn for a telecommunications company using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

## 📊 Project Overview

This project analyzes customer data from a telecommunications company to predict whether a customer is likely to churn (leave the service). The model uses various customer attributes such as tenure, monthly charges, service subscriptions, and contract details to make predictions.

## 🎯 Objective

Predict customer churn using a binary classification model to help the business:
- Identify customers at risk of leaving
- Implement retention strategies proactively
- Reduce customer acquisition costs
- Improve customer lifetime value

## 📁 Dataset

The dataset (`telco_data.csv`) contains information about:
- **Customer Demographics**: Gender, Partner, Dependents
- **Service Information**: Phone service, Internet service, Online security, Tech support, etc.
- **Account Information**: Contract type, Payment method, Tenure, Monthly charges, Total charges
- **Target Variable**: Churn (Yes/No)

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** & **Seaborn** - Data visualization
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Data preprocessing and model evaluation

## 🔍 Project Workflow

### 1. Data Exploration
- Load and inspect the dataset
- Identify data types and missing values
- Handle missing entries (removed rows with empty TotalCharges)

### 2. Data Visualization
- Analyzed churn patterns based on:
  - **Tenure**: Customers with shorter tenure show higher churn rates
  - **Monthly Charges**: Higher monthly charges correlate with increased churn

### 3. Data Preprocessing

#### Data Cleaning
- Removed `customerID` column (not useful for prediction)
- Handled missing values in `TotalCharges` field
- Standardized categorical values:
  - Replaced "No internet service" → "No"
  - Replaced "No phone service" → "No"

#### Feature Engineering
- **Binary Encoding**: Converted Yes/No columns to 1/0
  - Partner, Dependents, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, Churn
- **Gender Encoding**: Female=1, Male=0
- **One-Hot Encoding**: Applied to categorical columns
  - InternetService (DSL, Fiber optic, No)
  - Contract (Month-to-month, One year, Two year)
  - PaymentMethod (Electronic check, Mailed check, Bank transfer, Credit card)

#### Feature Scaling
- Applied MinMaxScaler to numerical features:
  - Tenure
  - MonthlyCharges
  - TotalCharges

### 4. Model Architecture

```
Input Layer:  26 neurons (26 features after preprocessing)
Hidden Layer: 15 neurons with ReLU activation
Output Layer: 1 neuron with Sigmoid activation (binary classification)
```

**Model Configuration:**
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Training Epochs**: 100
- **Train-Test Split**: 80-20

### 5. Model Evaluation

The model is evaluated using:
- **Accuracy**: Overall model performance
- **Confusion Matrix**: Visual representation of predictions vs. actual values
- **Classification Report**: Precision, Recall, F1-score for both classes
- **Class-specific Metrics**:
  - Precision for Churn (Class 1) and No Churn (Class 0)
  - Recall for Churn (Class 1) and No Churn (Class 0)

## 📈 Results

The model successfully predicts customer churn with:
- Binary classification output (0 = No Churn, 1 = Churn)
- Threshold of 0.5 for classification decision
- Detailed performance metrics for both classes

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/MathewX470/Deep-Learning-Projects.git
cd "Telco Customer Churn Prediction ANN"
```

2. Ensure `telco_data.csv` is in the project directory

3. Open and run the Jupyter notebook:
```bash
jupyter notebook churn.ipynb
```

## 📊 Visualizations

The project includes:
- **Histogram plots** showing churn distribution across:
  - Customer tenure
  - Monthly charges
- **Confusion matrix heatmap** for model predictions
the model as a web service

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**MathewX470**
- GitHub: [@MathewX470](https://github.com/MathewX470)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/MathewX470/Deep-Learning-Projects/issues).

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!
