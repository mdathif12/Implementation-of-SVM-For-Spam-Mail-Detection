# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### STEP 1 : Start
#### STEP 2 : Preprocessing the data
#### STEP 3 : Feature Extraction
#### STEP 4 : Training the SVM model
#### STEP 5 : Model Evalutaion
#### STEP 6 : Stop

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: Mohamed Athif Rahuman J
RegisterNumber:  212223220058

```
```
import pandas as pd
data = pd.read_csv("D:/introduction to ML/jupyter notebooks/spam.csv",encoding = 'windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x = data['v2'].values
y = data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.35,random_state = 48)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)
```

## Output:
#### Data:
![image](https://github.com/arbasil05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144218037/ac883980-2ee2-46fd-8cdb-82f067a463c1)
#### Data.shape:
![image](https://github.com/arbasil05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144218037/4b708069-05fb-481c-953a-1ccd957688e3)
#### x_train:
![image](https://github.com/arbasil05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144218037/083bd74b-bfa1-4d51-b55e-6c7fe12064f5)
#### Accuracy:
![image](https://github.com/arbasil05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144218037/ed95afb6-055d-478f-84b9-07d628180b3c)
#### Classification report:
![image](https://github.com/arbasil05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144218037/b699b079-ded5-42e7-8ad6-2688df23694f)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
