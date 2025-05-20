# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results. 

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Ravikrishnamoorthy D

Register Number: 212224040271

*/
```
import pandas as pd
df=pd.read_csv("Placement_Data.csv")
df.head()

data1=df.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

new_data = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
placement_status = model.predict(new_data)
print("Predicted Placement Status:", placement_status)

```

## Output:

![image](https://github.com/user-attachments/assets/fa66d743-bd04-49cb-a7bd-9960070b3398)

![image](https://github.com/user-attachments/assets/4d5b2bb3-aa9e-4cfb-8db4-d330c53cedf3)

![image](https://github.com/user-attachments/assets/a7f86a10-5523-4936-93bc-e9cb74fa9881)

![image](https://github.com/user-attachments/assets/00bcd6da-afdb-47a7-9b4f-d663a55277fc)

![image](https://github.com/user-attachments/assets/5f6ba294-639a-4d33-9398-b98f506f5141)

![image](https://github.com/user-attachments/assets/0227d5e8-c7c7-42f9-9d6b-069be3904b4e)

![image](https://github.com/user-attachments/assets/d8b27c79-e7dd-4b4a-ab83-adf301a9f2b3)

![image](https://github.com/user-attachments/assets/545be229-e962-4e8f-b5b7-cdc4f09f0eb9)

![image](https://github.com/user-attachments/assets/f14c2555-cf61-43b0-86d5-f7890e625406)

![image](https://github.com/user-attachments/assets/70a0e333-731f-4a5a-b4dc-d914d7e8ab62)

![image](https://github.com/user-attachments/assets/bb3f3a20-b26a-4310-95d2-b15c9630a928)

![image](https://github.com/user-attachments/assets/d67a48d8-8ad5-4754-9771-66c95fcf3986)














## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
