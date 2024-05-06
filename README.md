# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1:Start.

Step 2:Import the required libraries.

Step 3:Read the data frame using pandas.

Step 4:Get the information regarding the null values present in the dataframe. 

Step 5:Split the data into training and testing sets.

Step 6:Convert the text data into a numerical representation using CountVectorizer. 

Step 7:Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.

Step 8:Finally, evaluate the accuracy of the model.

Step 9:Stop. 

## Program:
```.
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Mohamed Athif Rahuman J
RegisterNumber: 212223220058

import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy 
*/
```

## Output:
## Result Output:
![image](https://github.com/Abinavsankar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119103734/583da52e-5fa6-4257-bc36-0bb606d3b023)

## data.head():
![image](https://github.com/Abinavsankar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119103734/b52c79de-8a0d-4627-816c-c852ccbc6468)

## data.info():
![image](https://github.com/Abinavsankar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119103734/7e723b14-22df-4021-b83c-568c1bbfc8d8)

## data.isnull.sum():
![image](https://github.com/Abinavsankar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119103734/334eac10-f78a-484f-9d50-29a99a98ee96)

## Y_prediction value:
![image](https://github.com/Abinavsankar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119103734/4ee49436-a061-4a34-9219-e1990be0955e)

## Accuracy value:
![image](https://github.com/Abinavsankar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119103734/b6a53556-d203-4a9b-8c96-f4bb98c970b6)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
