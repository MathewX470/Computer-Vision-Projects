# 🏦 Bank Customer Churn Prediction using ANN

A deep learning project that predicts customer churn for a bank using Artificial Neural Networks (ANN) with advanced techniques to handle class imbalance.

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Key Features](#key-features)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Learnings](#key-learnings)

## 🎯 Problem Statement

Customer churn is a critical issue for banks, as retaining existing customers is more cost-effective than acquiring new ones. This project aims to predict whether a customer will leave the bank (churn) based on various features such as credit score, geography, age, tenure, balance, and more.

### Challenge: Class Imbalance
The dataset exhibits significant class imbalance with approximately **80% non-churners** and **20% churners**, making it difficult for standard models to accurately predict the minority class (churners).

## 📊 Dataset

The dataset contains **10,000 customer records** with the following features:

- **CustomerId**: Unique identifier for each customer
- **Surname**: Customer's last name
- **CreditScore**: Credit score of the customer
- **Geography**: Country (France, Germany, Spain)
- **Gender**: Male or Female
- **Age**: Customer's age
- **Tenure**: Number of years with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of bank products used
- **HasCrCard**: Whether customer has a credit card (0/1)
- **IsActiveMember**: Whether customer is an active member (0/1)
- **EstimatedSalary**: Estimated annual salary
- **Exited**: Target variable (0 = Not Churned, 1 = Churned)

## 🔍 Approach

### 1. Data Preprocessing
- Removed irrelevant features (`CustomerId`, `RowNumber`, `Surname`)
- Encoded categorical variables:
  - **Gender**: Binary encoding (Male=0, Female=1)
  - **Geography**: One-Hot Encoding
- Split data into training (80%) and testing (20%) sets
- Applied **StandardScaler** for feature normalization

### 2. Exploratory Data Analysis
- Visualized churn patterns across different features (Tenure, Age)
- Identified class imbalance in the target variable

### 3. Model Architecture
Built a Sequential ANN with the following architecture:
```
Input Layer (11 features)
    ↓
Dense Layer (32 neurons, ReLU activation)
    ↓
Dropout (0.3)
    ↓
Dense Layer (16 neurons, ReLU activation)
    ↓
Dropout (0.3)
    ↓
Dense Layer (8 neurons, ReLU activation)
    ↓
Dropout (0.2)
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

### 4. Handling Class Imbalance
Implemented **two key techniques**:

#### a) Class Weights
- Calculated balanced class weights using `sklearn.utils.class_weight`
- Penalizes the model more for misclassifying minority class (churners)
- Applied during model training

#### b) Optimal Threshold Tuning
- Instead of default 0.5 threshold, found optimal threshold using Precision-Recall curve
- Maximizes F1-score to balance precision and recall
- Significantly improves recall for churners

## 📈 Results

### Original Model (Without Improvements)
```
Recall (Churn):    0.277 (only catching 28% of churners!)
Precision (Churn): 0.796
F1-Score (Churn):  0.411
```

### Improved Model (Class Weights + Optimal Threshold)
```
Recall (Churn):    0.621 (catching 62% of churners!)
Precision (Churn): 0.622
F1-Score (Churn):  0.622
Accuracy:          85%
```

### Confusion Matrix (Improved Model)
```
                Predicted
              No Churn  Churn
Actual No     1459      148
       Yes    149       244
```

### Key Improvements
- ✅ **Recall increased by ~124%** (0.277 → 0.621)
- ✅ **F1-Score increased by 51%** (0.411 → 0.622)
- ✅ **Better balance** between precision and recall
- ✅ **Catching 2x more churners** while maintaining reasonable precision

### Business Impact
- The model now identifies **62% of customers likely to churn**, enabling proactive retention strategies
- Balanced approach minimizes both missed churners and false alarms
- ROC-AUC score demonstrates strong discriminative ability

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Data preprocessing, metrics, class weights
- **Imbalanced-learn** - SMOTE (for comparison)

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/MathewX470/Deep-Learning-Projects.git
cd "Bank Customer Churn Prediction ANN"
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn
```

## 🚀 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook bank_churn_ANN.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the data
   - Train the original baseline model
   - Apply improved techniques (class weights + optimal threshold)
   - View comprehensive results and visualizations

3. The notebook includes:
   - Data exploration and visualization
   - Model training with class imbalance handling
   - Threshold optimization with Precision-Recall curves
   - ROC curve analysis
   - Performance comparison

## 📁 Project Structure

```
Bank Customer Churn Prediction ANN/
│
├── bank_churn_ANN.ipynb    # Main Jupyter notebook with complete analysis
├── data.csv                # Dataset (10,000 customer records)
└── README.md               # Project documentation
```

## 💡 Key Learnings

1. **Class Imbalance is Critical**: Standard metrics like accuracy can be misleading when dealing with imbalanced datasets

2. **Multiple Solutions Exist**: Tested various approaches:
   - Class Weights (✅ Best for this problem)
   - SMOTE (Synthetic oversampling)
   - Focal Loss (Custom loss function)
   - Threshold Tuning (✅ Combined with class weights)

3. **Business Context Matters**: For churn prediction, recall is more important than precision as missing actual churners is costlier than false alarms

4. **Threshold Tuning**: The default 0.5 classification threshold isn't always optimal, especially for imbalanced problems

5. **Visualization is Essential**: Precision-Recall and ROC curves provide valuable insights for threshold selection

## 📝 License

This project is part of a personal learning portfolio and is available for educational purposes.

⭐ If you found this project helpful, please consider giving it a star!
