from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.datasets import load_breast_cancer
import joblib
from MyGaussianNB import MyGaussianNB

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

my_gnb = MyGaussianNB()
my_gnb.fit(X_train, y_train)
my_gnb.predict(X_test)
my_gnb.score(X=X_test, y=y_test)
mode = joblib.dump(filename="GaussianNB.model", value=my_gnb)