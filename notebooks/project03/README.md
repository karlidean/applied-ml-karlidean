# P3 - Building a Classifier

**Author:** Karli Dean  
**Date:** November 7, 2025  
**Course:** Applied Machine Learning / Business Intelligence & Analytics

## Overview
This project develops and evaluates multiple classification models to predict passenger survival. The work is completed in a structured Jupyter notebook that follows the full lifecycle of building an applied machine learning classifier: data preparation, exploratory analysis, feature engineering, model training, model evaluation, and comparative results analysis.

## Objective
The primary objective is to compare the performance of several model types under different feature configurations in order to understand how feature selection affects predictive performance. Additionally, this project demonstrates core machine learning workflow practices, including preprocessing, model tuning considerations, and evaluation using test-set performance metrics.

## Dataset
The dataset represents individual passenger records including demographic, ticket, and family information, along with a binary survival outcome. Key preprocessing steps included:
- Handling missing values
- Standardizing structured fields
- Encoding categorical attributes
- Deriving engineered features (e.g., family size)

## Modeling Approach
Three feature cases were examined:

| Case | Features Used | Description |
|------|--------------|-------------|
| **Case 1** | Single feature | Baseline prediction using one feature for comparison |
| **Case 2** | Age | Numerical demographic predictor |
| **Case 3** | Age + Family Size | Combination of demographic and relational attributes |

For each case, multiple algorithms were evaluated:

- **Decision Tree Classifier**
- **Support Vector Machine (RBF Kernel)**
- **Neural Network (Multi-Layer Perceptron)**, applied to Case 3 after scaling

## Evaluation Metrics
Model performance was assessed using standard binary classification metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

These metrics were computed on a **held-out test set** to ensure fair evaluation.

## Results Summary
Results are compiled into a performance comparison table within the notebook, enabling cross-model and cross-case interpretation. In general:

- Case 3 consistently performed better than Case 1 and Case 2, indicating the importance of feature selection.
- The Neural Network model in Case 3 demonstrated competitive performance after scaling inputs.
- Performance varied notably by model type, demonstrating tradeoffs in interpretability vs. predictive strength.

## Key Findings
- Model quality improves with richer, well-engineered input features.
- Linear models struggle when decision boundaries are non-linear; kernel-based SVMs and neural networks perform better in such conditions.
- Decision Trees provide interpretability advantages but may generalize less effectively without pruning / tuning.

## Repository Structure
