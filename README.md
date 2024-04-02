# EXP NO: 6-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
```

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: POPURI SRAVANI
RegisterNumber:  212223240117
*/
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
## DATASET
![Screenshot 2024-04-02 132134](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139778301/49e03fe8-3fba-41eb-951f-917157c19969)
## data.info()
![Screenshot 2024-04-02 132141](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139778301/69fb0f07-e2c5-4912-8249-8d39130a1716)
## CHECKING IF NULL VALUES ARE PRESENT
![Screenshot 2024-04-02 132151](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139778301/148dbe24-d7d2-445b-8e9d-6bdb2ce1de07)
## VALUE_COUNTS()
![Screenshot 2024-04-02 132200](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139778301/77ea26d4-438d-4b37-bcd2-31111ecd8ffa)
## DATASET AFTER ENCODING
![Screenshot 2024-04-02 132210](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139778301/b3570640-c06d-4f8e-aa6b-70b97c735f19)
## X-VALUES
![Screenshot 2024-04-02 132219](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139778301/64f673aa-9346-4d1e-98e4-26ae8c7de534)
## ACCURACY
![Screenshot 2024-04-02 132227](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139778301/90074a6b-4a52-402c-9f1c-3943bc6a3bea)
## DT.PREDICT()
![Screenshot 2024-04-02 132238](https://github.com/sravanipopuri2006/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/139778301/02b00c3f-a6c4-4af8-a004-f2ed97eef485)










## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
