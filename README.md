# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets

2.Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters

3.rain your model -Fit model to training data -Calculate mean salary value for each subset

4.Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance

5.Tune hyperparameters -Experiment with different hyperparameters to improve performance

6.Deploy your model Use model to make predictions on new data in real-world application.

## Program:

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: mahalakshmi.k

RegisterNumber:212222240057  
*/


import pandas as pd

df=pd.read_csv('/content/Salary.csv')


df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df["Position"]=le.fit_transform(df["Position"])

df.head()

x=df[["Position","Level"]]

y=df["Salary"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])

*/

## Output:

1.data.head()

![1 data head()](https://github.com/maha712/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121156360/38026efa-619d-4447-acde-4fdf010704ea)

2.data.info()

![2 data info()](https://github.com/maha712/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121156360/f9717175-ef4f-4932-bcd3-fea534d36640)

3.data.isnull().sum()

![3 data isnull() sum()](https://github.com/maha712/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121156360/157a6b71-c03d-413a-8b0d-975c169bbcc8)

4.data.head() for position:

![4 data head()](https://github.com/maha712/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121156360/3ce2ec21-cab7-4efd-a7ce-4ed3c3f8027a)

5.MSE value:

![5 MSE](https://github.com/maha712/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121156360/012e6ac8-4f59-4ec0-a862-8d1479e2a94c)

6.R2 value:

![6 R2 value](https://github.com/maha712/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121156360/bbd95462-9b9b-408c-bea0-4c9bfca9ae5b)

7.Prediction Value:

![7 Prediction Value](https://github.com/maha712/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121156360/bec6225c-be47-43a2-a85f-4fcb75bbb9bf)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
