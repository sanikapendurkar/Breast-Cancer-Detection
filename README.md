# Breast Cancer Prediction with Various Machine Learning Models

This project demonstrates how to use various machine learning models to predict breast cancer based on the **Coimbra Breast Cancer Dataset**. The dataset contains medical data with features such as `Age`, `BMI`, `Glucose`, `Insulin`, etc., and the task is to classify patients as either `healthy` (0) or having `cancer` (1).

## Overview

In this repository, we explore multiple machine learning models and techniques to classify breast cancer. We also address class imbalance by applying data resampling techniques such as **SMOTE** (Synthetic Minority Over-sampling Technique) and **ADASYN** (Adaptive Synthetic Sampling). The models evaluated in this project include:

- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree Classifier**

We compare the performance of these models both with and without resampling techniques, and assess the models using various performance metrics like **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **AUC-ROC**.

## Dataset

The **Coimbra Breast Cancer Dataset** contains the following features:

- **Age**
- **BMI (Body Mass Index)**
- **Glucose**
- **Insulin**
- **HOMA (Homeostasis Model Assessment)**
- **Leptin**
- **Adiponectin**
- **Resistin**
- **MCP.1**

The target variable is `Classification`, where:

- `0` represents healthy (non-cancer)
- `1` represents cancer

The dataset is used for binary classification, where the goal is to predict whether a patient is healthy or has cancer.

## Project Structure

The repository contains several Python scripts that perform different tasks related to training models, evaluating their performance, and comparing results. Below are the scripts included in the repository:

### Scripts

1. **`adasyn.py`**  
   This script applies the **ADASYN** resampling technique to balance the dataset, trains multiple machine learning models (SVM, Logistic Regression, Random Forest, Decision Tree), and evaluates their performance.

2. **`adasyn_comb.py`**  
   Similar to `adasyn.py`, but this script compares the performance of the models using ADASYN and visualizes the results using ROC curves.

3. **`originaldataset.py`**  
   This script trains and evaluates the models on the **original dataset** (without any resampling) and visualizes the ROC curves for each model.

4. **`originaldataset_comb.py`**  
   An extended version of `originaldataset.py`, this script compares multiple models (SVM, Logistic Regression, Random Forest, Decision Tree) on the original dataset, without oversampling, and plots their ROC curves.

5. **`smote.py`**  
   This script applies the **SMOTE** technique to balance the dataset and evaluates the performance of different models (SVM, Logistic Regression, Random Forest, Decision Tree).

6. **`smote_comb.py`**  
   Similar to `smote.py`, this script compares the performance of multiple models using SMOTE for balancing the dataset and visualizes the ROC curves for each model.

### Key Features

- **SMOTE and ADASYN**: Both scripts using these techniques apply resampling to handle imbalanced datasets, which helps improve the performance of classifiers on minority classes.
  
- **Model Evaluation**: Each script evaluates models using metrics like **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **AUC-ROC**.
  
- **ROC Curve**: The scripts also generate and compare **ROC curves** to visualize how well each model distinguishes between the two classes (healthy vs cancer).

## Dependencies

Ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `imbalanced-learn`

To install the required libraries, use the following command:

```bash
pip install numpy pandas scikit-learn matplotlib imbalanced-learn
```

## Usage

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/Coimbra-Breast-Cancer-Classification.git
cd Coimbra-Breast-Cancer-Classification
```

### Running the Scripts

After cloning the repository and navigating to the project directory, you can run any of the provided scripts. Below are examples of how to run each script:

1. **Run ADASYN resampling script**:
   ```bash
   python adasyn.py
   ```

2. **Run ADASYN combined results script (comparison and ROC curve)**:
   ```bash
   python adasyn_comb.py
   ```
   
3. **Run Original dataset script (without resampling)**:
   ```bash
   python originaldataset.py
   ```
   
4. **Run Original dataset combined results script**:
   ```bash
   python originaldataset_comb.py
   ```
   
5. **Run SMOTE resampling script**:
   ```bash
   python smote.py
   ```
   
6. **Run SMOTE combined results script (comparison and ROC curve)**:
   ```bash
   python smote_comb.py
   ```
   
## Example of Expected Output

Each script will output the following evaluation metrics for the models being evaluated:

For each model, you will see a summary similar to this:

### SVM Model Evaluation:
```plaintext
Accuracy: 0.91
Precision: 0.85
Recall: 0.94
F1-Score: 0.89
AUC-ROC: 0.93
```

### Random Forest Model Evaluation:
```plaintext
Accuracy: 0.92
Precision: 0.88
Recall: 0.94
F1-Score: 0.91
AUC-ROC: 0.95
```

### Logistic Regression Model Evaluation:
```plaintext
Accuracy: 0.90
Precision: 0.83
Recall: 0.93
F1-Score: 0.88
AUC-ROC: 0.92
```

### Decision Tree Model Evaluation:
```plaintext
Accuracy: 0.87
Precision: 0.80
Recall: 0.91
F1-Score: 0.85
AUC-ROC: 0.89
```

## Model Evaluation Metrics

Each script evaluates the models using the following metrics:

- **Accuracy**: The percentage of correct predictions made by the model.
- **Precision**: The percentage of positive predictions that were correct.
- **Recall**: The percentage of actual positives that were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.
- **AUC-ROC**: The area under the Receiver Operating Characteristic curve. A higher AUC indicates better performance in distinguishing between the positive and negative classes.

## ROC Curves

Each script generates **ROC curves** for the models, which help visualize how well each model distinguishes between the two classes (healthy vs cancer). The **AUC-ROC** score quantifies the performance, where higher values (closer to 1) indicate better classification performance.

## Results Comparison

After running the scripts, you will see the model performance across different techniques and resampling methods. The **AUC-ROC** scores are particularly useful for comparing the performance of the classifiers.

### Example: Comparison of Models with Resampling (SMOTE and ADASYN)

| Model           | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-----------------|----------|-----------|--------|----------|---------|
| SVM             | 0.91     | 0.85      | 0.94   | 0.89     | 0.93    |
| Random Forest   | 0.92     | 0.88      | 0.94   | 0.91     | 0.95    |
| Logistic Reg.   | 0.90     | 0.83      | 0.93   | 0.88     | 0.92    |
| Decision Tree   | 0.87     | 0.80      | 0.91   | 0.85     | 0.89    |

### Example: Comparison of Models without Resampling (Original Dataset)

| Model           | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-----------------|----------|-----------|--------|----------|---------|
| SVM             | 0.89     | 0.84      | 0.91   | 0.87     | 0.91    |
| Random Forest   | 0.90     | 0.86      | 0.92   | 0.89     | 0.93    |
| Logistic Reg.   | 0.88     | 0.81      | 0.90   | 0.85     | 0.90    |
| Decision Tree   | 0.85     | 0.78      | 0.88   | 0.83     | 0.87    |

## Conclusion

This project demonstrates the effectiveness of different machine learning models for breast cancer classification, both with and without resampling techniques. The results show that using **SMOTE** or **ADASYN** can significantly improve model performance, particularly in terms of recall and AUC-ROC.
