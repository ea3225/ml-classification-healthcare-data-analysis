# Machine Learning Classification for Healthcare Data Analysis

## Overview

This project applies supervised machine learning methods to a healthcare classification task using the Cardio Train dataset. The objective is to predict whether a patient is likely to have cardiovascular disease based on demographic, behavioral, and clinical health indicators.

The notebook covers the full classification workflow, including data preprocessing, manual implementation of logistic regression, model training with scikit-learn, gradient boosting, ROC/AUC analysis, and cost-based decision threshold selection.

## Dataset

The project uses the `cardio_train` dataset, a structured healthcare dataset containing patient-level features related to cardiovascular risk.

Features include variables such as:

* age
* gender
* height
* weight
* diastolic blood pressure
* cholesterol
* glucose
* smoking status
* alcohol use
* physical activity

The target variable is:

* `cardio`: indicator of cardiovascular disease presence

As part of preprocessing, the `ap_hi` feature was removed in the lab setup.

## Methodology

### Data Preprocessing

* Loaded the healthcare dataset from CSV format
* Removed the `ap_hi` feature
* Split the data into training and test sets
* Standardized features where needed for logistic regression training

### Logistic Regression

The first part of the project implements logistic regression manually using:

* sigmoid activation
* log-loss computation
* gradient descent optimization

This manual implementation is then compared against scikit-learn's `LogisticRegression` model.

### Gradient Boosting

The notebook also applies gradient boosting classifiers using two loss functions:

* log-loss
* exponential loss

A grid search is used to tune the learning rate and identify the best-performing configuration.

### Decision Analysis

Beyond standard classification metrics, the project evaluates the model from a decision-making perspective by:

* computing ROC and AUC
* analyzing classification thresholds
* minimizing expected screening cost under different cost assumptions for false positives and false negatives

### Optional Extension

An additional section explores Support Vector Machines with different kernels:

* RBF
* linear
* sigmoid

## Models Implemented

* Manual Logistic Regression
* Scikit-learn Logistic Regression
* Gradient Boosting Classifier
* Support Vector Machine

## Results

Some key outcomes from the notebook include:

* Manual logistic regression achieved about **65.5% accuracy**
* Including the removed feature improved logistic regression accuracy to about **67.6%**, and a longer training run reached about **71.9%**
* Scikit-learn logistic regression achieved about **65.5% accuracy** on the reduced-feature dataset
* Gradient boosting achieved about **70.7% to 70.8% accuracy**
* Hyperparameter tuning identified **0.15** as the best learning rate for the tested gradient boosting setup
* In the optional SVM section:

  * RBF kernel achieved about **60.3% accuracy**
  * linear kernel achieved about **65.1% accuracy**
  * sigmoid kernel performed substantially worse

The notebook also identifies an approximately optimal probability threshold of **0.2626** in one decision-cost scenario.

## Key Insights

* Gradient boosting outperformed the baseline logistic regression models on this dataset
* Feature selection had a noticeable impact on model performance
* Increasing training iterations improved the manually implemented logistic regression model
* Model evaluation should not rely only on accuracy; threshold selection and cost-sensitive analysis can materially change the preferred decision rule
* In healthcare settings, predictive performance and decision cost must be considered together

## Repository Structure

```text
ml-classification-healthcare-data-analysis/
│
├── data/
│   └── cardio_train.csv
├── model/
│   └── ml_classification_model.ipynb
├── README.md
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook and run the cells in order.

## Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Notes

* This project was completed as an individual machine learning lab
* The notebook emphasizes both model implementation and model evaluation
* The focus is on healthcare classification, predictive performance, and decision-aware analysis
