# AI Data Analysis and Prediction  

## Project Overview  

This project applies machine learning techniques to analyze and predict outcomes based on different datasets, covering both classification and regression problems. Various visualization techniques are used to explore data characteristics.  

---

## 1. Wine Data Analysis and Classification Prediction  

### **Objective**  

- Read the wine dataset and encode the wine type column using **LabelEncoder**.  
- Test three machine learning algorithms (**K-Nearest Neighbors (KNN), Decision Tree, and Random Forest**) and identify the model with the highest accuracy.  

### **Dataset**  

- **Source:** Wine Data  
- **Target Variable:** `wine.target` (Three wine types: **Barolo, Grignolino, Barbera**)  

### **Methods and Steps**  

1. Read the dataset and apply **Label Encoding**.  
2. Split the dataset into **training and testing sets** using `train_test_split`.  
3. Train and evaluate three machine learning models:  
   - **KNN (K-Nearest Neighbors)**  
   - **Decision Tree**  
   - **Random Forest**  
4. Calculate the accuracy of each model and identify the best-performing one.  

### **Results**  

| Model         | Accuracy (%) |
|--------------|-------------|
| KNN          | 72.2%       |
| Decision Tree | 94.4%       |
| **Random Forest** ✅ | **100%** |

**Best Algorithm:** **Random Forest**  

---

## 2. Italian Wine Data Analysis  

### **Objective**  

Analyze 13 different features influencing **Italian wine classification**, generate a **heatmap** and **pair plots** to identify the most significant features for classification.  

### **Dataset**  

- **Source:** Italian Wine Data  
- **Target Variable:** Wine Type (**0: Barolo, 1: Grignolino, 2: Barbera**)  

### **Methods and Steps**  

1. Read the dataset and compute **correlation coefficients**.  
2. Select features with **absolute correlation values greater than 0.7** with the wine type.  
3. Generate a **heatmap** (correlation matrix) and **pair plot** for selected features.  
4. Perform **wine type distribution analysis** using pie charts.  

### **Results**  

- **Most significant features for classification:** **Total Phenols, Flavonoids, Diluted Wine**  
- **Most dominant wine type in dataset:** **Barolo**  

---

## 3. Salary Prediction (Regression Analysis)  

### **Objective**  

Use machine learning to predict salaries and evaluate the model’s performance for **overfitting** detection.  

### **Dataset**  

- **Source:** Salary Data  
- **Target Variable:** Salary  

### **Methods and Steps**  

1. Read the dataset and remove irrelevant columns.  
2. Split the dataset into **training and testing sets** using `train_test_split`.  
3. Train a **Linear Regression** model.  
4. Evaluate model performance using **R-squared (R²)**.  
5. Compare **actual vs. predicted salary values** using visualizations.  

### **Results**  

| Data Type  | R² Score |
|------------|---------|
| Training Data | 0.88 |
| Test Data     | 0.89 |

- **No Overfitting:** Test accuracy is similar to training accuracy.  

---

## 4. Customer Purchase Prediction (Classification)  

### **Objective**  

Use machine learning to predict whether a customer will purchase a product and evaluate the classification model’s effectiveness.  

### **Dataset**  

- **Source:** Customer Purchase Data  
- **Target Variable:** **Purchased** (0 = Not Purchased, 1 = Purchased)  

### **Methods and Steps**  

1. Convert **categorical variables** using **LabelEncoder** (`Gender`).  
2. Analyze data correlation and identify variables with the highest correlation to purchasing behavior.  
3. Train a **Decision Tree** classifier.  
4. Generate a **Confusion Matrix** to evaluate classification performance.  
5. Compute model **accuracy score**.  

### **Results**  

| Data Type  | Accuracy (%) |
|------------|-------------|
| Training Data | 99.69%     |
| Test Data     | 85.00%     |

- **No Overfitting:** The test accuracy remains relatively high.  

---

## **Technologies and Tools Used**  

- **Programming Language:** Python  
- **Machine Learning Library:** Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Visualization Tools:** Matplotlib, Seaborn  

---

## **How to Run the Program**  

1. Install the required packages:  

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Download the datasets and place them in the appropriate directory.  
3. Run the script:  

   ```bash
   python script.py
   ```

---

## **Project Conclusion**  

This project applies **machine learning techniques** to analyze different datasets and derives the following key insights:  

1. **Random Forest** performed best in wine classification, achieving **100% accuracy**.  
2. **Alcohol content, total phenols, and flavonoids** significantly influence wine classification.  
3. **Training hours have a strong linear correlation with salary**.  
4. **Decision Tree** effectively predicts customer purchase behavior but requires further optimization to avoid overfitting.  

---
