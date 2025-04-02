### Introduction

This repository contains the work completed during my **Introduction to Machine Learning** course in the spring semester of 2025. It includes both independent study tasks and laboratory exercises that helped deepen my understanding of machine learning concepts. These tasks provided hands-on experience with key machine learning algorithms and data processing techniques, reinforcing the theoretical knowledge learned in class.

*Tools:* Pandas, Scikit-learn, Matplotlib, Seaborn

The course covered the following main topics according to the syllabus:

- **Introduction to Machine Learning**: Basics of machine learning, applications, and Python libraries for data analysis.
- **Data and Datasets**: Types of data (tabular, image, text) and data mining techniques.
- **Supervised Learning**: Understanding models, loss functions, and errors like overfitting and underfitting.
- **Linear Algebra**: Concepts of vectors, matrices, matrix operations, and rank calculations.
- **Linear Regression**: Foundations of simple and multiple linear regression, polynomial regression, and model evaluation.
- **Optimization**: Derivatives, gradients, and methods like gradient descent.
- **Logistic Regression**: Classification, cost functions, and regularization.
- **Probability and Statistics**: Key probability concepts and the central limit theorem.
- **Support Vector Machines (SVM) & K-Nearest Neighbors (KNN)**: Theoretical foundations and practical applications of these algorithms.

### Summary of Exercises

- **Data Preprocessing with Scikit-learn**:
    - Learned to **encode categorical variables** using `OneHotEncoder` and `LabelEncoder`.
    - Gained experience in feature scaling with `MinMaxScaler` and `StandardScaler` .
- **Gradient Boosting Model**:
    - Built and trained a `GradientBoostingRegressor` model.
    - Focused on optimizing the `n_estimators` and `learning_rate` hyperparameters.
    - Evaluated the model’s performance using **Mean Squared Error (MSE)**, experimenting with different parameter settings to minimize overfitting and enhance accuracy.
- **Gaussian Elimination for Rank Calculation**:
    - Implemented Gaussian elimination to calculate the rank of a matrix and test for linear independence of its columns, reinforcing the importance of matrix operations and row echelon form.
- **Linear Regression**:
    - Created a custom linear regression model using **Ordinary Least Squares (OLS)** for manual parameter estimation, improving understanding of regression theory, including coefficient and intercept estimation.
- **Residual Analysis in Linear Regression**:
    - Learned to calculate and interpret residuals to assess how well the linear regression model fits the data.
    - Analyzed residual plots to identify model errors and determine when linear models might be inappropriate.
- **Polynomial Regression**:
    - Explored the concept of polynomial regression to model non-linear relationships between variables.
    - Transformed input features into higher-degree polynomials, observing how it improves model accuracy when linear relationships do not fit the data well.
- **K-Nearest Neighbors (KNN) Algorithm**:
    - Implemented the KNN algorithm and visualized how different values of `k` affected model predictions.
- **Complex Datasets and Metrics in KNN**:
    - Generated complex datasets using `make_blobs` and tested KNN performance with both Euclidean and Manhattan distances.
    - Evaluated the model's performance using metrics such as **accuracy,** **precision, and recall** to assess how well the classifier performed on the test data.
- **SVM Implementation**:
    - Implemented a linear Support Vector Machine (SVM) model to classify data using `sklearn.svm.SVC`.
    - Visualized the **decision boundary**, identified **support vectors**, and analyzed the model’s behavior for linearly separable data.
- **Soft Margin SVM & Slack Variables**:
    - Incorporated **slack variables** into the SVM to handle misclassifications for non-linearly separable data.
    - Experimented with different values of the **regularization parameter** `C` to control the trade-off between margin size and classification error, adjusting model complexity accordingly.

This collection of exercises allowed me to build a stronger foundation in machine learning, with a focus on key techniques for both supervised learning tasks (regression and classification) and model evaluation. By implementing these algorithms and exploring different datasets, I gained practical experience in preprocessing data, building models, and evaluating their performance.
