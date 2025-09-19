# 📘 Machine Learning Foundations

This repository contains my implementations and exercises from the **Holberton School Machine Learning specialization**.  
The goal of this repo is to build a solid understanding of **machine learning concepts**, focusing on implementing algorithms from scratch in Python.

---

## 🧠 Repository Structure

- 📂 **math** → Linear algebra, probability, and calculus basics for ML  
- 📂 **supervised_learning** → Linear/logistic regression, decision trees, random forests, neural networks  
- 📂 **unsupervised_learning** → Clustering (K-means, hierarchical), PCA, dimensionality reduction  
- 📂 **reinforcement_learning** → Q-learning, exploration vs. exploitation  
- 📂 **pipeline** → Data preprocessing, feature engineering, and evaluation  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn  
- **Tools:** Jupyter Notebooks  

---

## 📊 Example: Logistic Regression
```python
from logistic_regression import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Accuracy:", model.evaluate(y_test, preds))
