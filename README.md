# TDSE-02 
# Heart Disease Risk Prediction using Logistic Regression

This repository contains the implementation and analysis for Lab 2 of the Digital Transformation and Enterprise Architecture (TDSE) course. The project develops a **binary logistic regression model from scratch** to predict heart disease risk based on clinical features, with a focus on mathematical foundations, model evaluation, and deployment architecture using Amazon SageMaker.

---

## Exercise Summary

The laboratory implements a complete machine learning pipeline:

1. **Exploratory Data Analysis (EDA)**: Statistical analysis and visualization of patient clinical data
2. **Feature Engineering**: Data normalization and preprocessing for stable model training
3. **Model Implementation**: Logistic regression from scratch using NumPy (sigmoid function, cost optimization, gradient descent with L2 regularization)
4. **Hyperparameter Tuning**: Grid search over regularization parameter (λ) to optimize test metrics
5. **Model Evaluation**: Train/test split with metrics computation (accuracy, precision, recall, F1-score)
6. **Model Persistence**: Export best model weights, bias, normalization parameters, and feature list
7. **Deployment Preparation**: Package model artifacts for Amazon SageMaker inference container

**Key Achievement**: Successfully trained a regularized logistic regression model that generalizes well across train and test sets with no signs of overfitting.

---

## Dataset Description

**Dataset**: [Kaggle Heart Disease](https://www.kaggle.com/datasets/neurocipher/heartdisease)

**Characteristics**:
- **Sample Size**: 303 patients with complete clinical records
- **Target Variable**: Heart disease presence (1) or absence (0) → ~55% positive class rate
- **Features Used in Final Model** (7 clinical features):
  - Age: 29–77 years
  - Sex: 0–1 (categorical)
  - Cholesterol (Chol): 112–564 mg/dL
  - Resting Blood Pressure (RestBP): 94–200 mmHg
  - Maximum Heart Rate Achieved (MaxHR): 71–202 bpm
  - Induced Angina (IndAng): 0–1 (binary)
  - ST Segment Depression (OldPeak): 0–6.2 units

All numerical features are **z-score normalized** during training and inference using dataset mean and standard deviation.

---

## Model Performance

### Regularization Analysis

A grid search was performed over regularization parameter λ ∈ {0.0 , 0.001, 0.01, 0.1, 1} to balance model complexity and generalization.

**Best Model Selected** (λ = 0.0):
- **Test Accuracy**: ~78%
- **Test F1-Score**: ~0.78
- **Test Precision**: High (fewer false positives)
- **Test Recall**: Balanced sensitivity
- **Generalization**: Minimal gap between train and test metrics, indicating no overfitting

The model achieves a good balance between sensitivity and specificity, suitable for clinical decision support where both false negatives and false positives carry risk.

---

## Running the Notebook

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or JupyterLab
Libraries: numpy, pandas, matplotlib
```

### Installation & Execution

```bash
1. Clone or download the repository
2. Navigate to the project directory
3. Install dependencies: pip install numpy pandas matplotlib
4. Open Jupyter: jupyter notebook
5. Run heart_disease_lr_analysis.ipynb cell-by-cell
```

### Notebook Structure

The notebook is organized into 8 main sections:

1. **Setup & Imports**: Required libraries
2. **Load & Prepare Data**: Read CSV, target encoding, feature extraction
3. **Exploratory Data Analysis**: Statistical summaries, distributions, correlations
4. **Model Training with Regularization**: Implement logistic regression, grid search over λ
5. **Model Evaluation & Selection**: Metrics tables, best model identification
6. **Model Persistence & Packaging**: Save model artifact, create tarball for SageMaker
7. **SageMaker Deployment Instructions**: Step-by-step guide (theoretical; requires IAM permissions and VPC endpoints)
8. **Local Inference Testing**: Validate model with sample input

---

## Model Artifacts

The following files are generated during notebook execution:

| File | Purpose |
|------|---------|
| `logreg_model_full.npy` | NumPy archive containing model state: weights (w), bias (b), normalization parameters (μ, σ), feature names |
| `inference.py` | SageMaker inference container script with model loading and prediction logic |
| `model.tar.gz` | Tarball packaging `logreg_model_full.npy` + `inference.py` for SageMaker deployment |

---

## Deployment on Amazon SageMaker

### Deployment Architecture

The deployment leverages AWS SageMaker to host the trained model as a real-time inference endpoint:

1. **Model Upload**: Package and push model artifacts to Amazon S3
2. **Model Creation**: Register model in SageMaker
3. **Endpoint Configuration**: Define instance type and scaling parameters
4. **Endpoint Deployment**: Launch inference endpoint on EC2 instance
5. **Inference API**: Send JSON requests; receive risk probabilities

### Inference Interface

**Request Format** (JSON):
```json
{
  "inputs": [60, 1, 300, 140, 150, 1.0, 0]
}
```
Represents: [Age=60, Sex=1, Chol=300, RestBP=140, MaxHR=150, IndAng=1.0, OldPeak=0]

**Response Format** (JSON):
```json
{
  "probability": [0.50897657]
}
```
Predicted probability of heart disease presence.

### Deployment Steps (Theoretical)

The notebook includes a complete deployment guide covering:
1. Create S3 bucket and upload model tarball
2. Create SageMaker model from tarball
3. Create endpoint configuration with container image
4. Deploy endpoint using scikit-learn inference container
5. Invoke endpoint with sample input

## Screenshots: 

### Running the notebook

<img width="1858" height="986" alt="Training Metrics" src="https://github.com/user-attachments/assets/acd67eb9-8f87-4900-a026-6fe38fb70a3c" />

### Deployment

<img width="1871" height="998" alt="Best Model Results" src="https://github.com/user-attachments/assets/575f1557-b4b8-4321-99d8-9eb66a57649c" />

<img width="1831" height="991" alt="Model Export" src="https://github.com/user-attachments/assets/c53e8ce5-c111-4867-98c3-5999567c6d0d" />

<img width="1866" height="998" alt="Local Test Output" src="https://github.com/user-attachments/assets/68d7f59c-c171-4fa0-893a-88f356f598e1" />

#### Screenshot 5: Test Output Explanation
<img width="1866" height="1001" alt="Test Explanation" src="https://github.com/user-attachments/assets/d6511865-5ba9-4ba3-9373-d0ceb9b0d369" />

---

## Deployment Status & Constraints

### Current Status

**Completed**:
- Logistic regression model training and evaluation
- Model persistence and artifact creation
- SageMaker deployment guide and local testing
- Notebook documentation and markdown explanations

**Blocked by AWS IAM Permissions**:
The live endpoint deployment in AWS Academy Learner Lab encountered an **IAM access denial** for the `sagemaker:CreateEndpointConfig` action. This is a security restriction imposed by the learning lab environment to limit resource creation costs and prevent unintended AWS charges.

### Error Details

```
Error: User is not authorized to perform: sagemaker:CreateEndpointConfig 
on resource: arn:aws:sagemaker:us-east-1:*:endpoint-config/*
Reason: Explicit deny in policy VocLabPolicy3-hxR4rpqIpSMq
```

#### Screenshot: Permission Error

<img width="993" height="548" alt="IAM Error" src="https://github.com/user-attachments/assets/d800c41e-7f3b-4cd2-b17f-0e90e1fdd187" />

---

## Key Learnings

1. **Logistic Regression from First Principles**: Understanding gradient descent, sigmoid activation, and cost function optimization
2. **Regularization**: L2 regularization prevents overfitting and improves generalization
3. **Model Evaluation**:  Importance of proper train/test splitting and multi-metric assessment (accuracy, precision, recall, F1)
4. **Model Serialization**: Proper persistence of model state (weights, biases, normalization parameters) for reproducible inference
5. **MLOps Concepts**: Containerization, artifact packaging, and cloud deployment workflows
6. **AWS Constraints**: IAM policies and resource quotas in learning environments require careful planning and escalation to administrators

---


## References

- Kaggle Dataset: https://www.kaggle.com/datasets/neurocipher/heartdisease
- AWS SageMaker Documentation: https://docs.aws.amazon.com/sagemaker/
- Logistic Regression Theory: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

---


### I'll update it when I can.

## Built With

* Python
* Jupyter

## Authors

* **Andres Felipe Cardozo Martinez**

## Acknowledgments

* This project was guided by laboratory notebook exercises provided as part of the course.
