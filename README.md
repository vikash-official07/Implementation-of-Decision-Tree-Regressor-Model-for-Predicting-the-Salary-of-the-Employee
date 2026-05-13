# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries
2. Load the employee salary dataset using pandas.
3. Select the input feature (X) and target variable (y)
4. Convert the selected columns into numpy arrays if required.
5. Create the Decision Tree Regressor model using DecisionTreeRegressor().
6. Train the model using the fit() method with X and y values.
7. Predict the salary using the predict() method for test input values.
8. Visualize the Decision Tree Regression model using matplotlib graph
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: vikash.s
RegisterNumber:  212225040490
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
data = pd.read_csv("Salary (1).csv")
X = data.drop("Salary", axis=1)
y = data["Salary"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
plt.figure(figsize=(25,12))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.title("Decision Tree Regressor")
plt.show()


```

## Output:
<img width="1384" height="619" alt="image" src="https://github.com/user-attachments/assets/719feddb-29f0-4a61-92bc-07dfdf33f799" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
