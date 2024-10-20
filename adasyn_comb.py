import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN  # Using ADASYN for oversampling

# Load dataset
df = pd.read_csv('Coimbra_Breast_Cancer_Dataset.csv')

# Define the feature columns and target column
features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']
target = 'Classification'

# Convert target labels 1 = healthy, 2 = cancer
df['Classification'] = df['Classification'].apply(lambda x: 0 if x == 1 else 1)  # 0 for healthy, 1 for cancer

# Splitting the data into features and target
X = df[features]
y = df[target]

# Splitting the data before applying ADASYN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply ADASYN
adasyn = ADASYN(random_state=42)
X_train, y_train = adasyn.fit_resample(X_train, y_train)

# Initialize and train models
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Function to plot ROC Curves for all models
def plot_comparison_roc(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models.items():
        # Get the predicted probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate the ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Plotting the diagonal line for random guessing
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ADASYN Comparison of ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

# Dictionary of models to compare
models = {
    'SVM': svm_model,
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'Decision Tree': dt_model
}

# Plot the ROC curves for all models
plot_comparison_roc(models, X_test, y_test)
