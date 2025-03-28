# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data 
2. Split Dataset into Training and Testing Sets.
3. Train the Model Using Stochastic Gradient Descent (SGD) 
4. Make Predictions and Evaluate Accuracy
5. Generate Confusion Matrix

## Program & Output:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: TAMIZHSELVAN B
RegisterNumber:  212223230225
*/
```

```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
iris=load_iris()
```

```
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
```

![EX_7_OUTPUT_1](https://github.com/user-attachments/assets/40846535-5ea5-480e-9fa5-3ed54f2fbbe3)

```
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)
```
```
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

![EX_7_OUTPUT_2](https://github.com/user-attachments/assets/4fb138e3-3208-4ad1-a315-b9e5b6f2b600)

```
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```

![EX_7_OUTPUT_3](https://github.com/user-attachments/assets/e7265e59-fcaa-404d-a203-dac6ed195398)

```
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```

![EX_7_OUTPUT_4](https://github.com/user-attachments/assets/b0665782-8183-42fe-89c5-455179b314df)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
