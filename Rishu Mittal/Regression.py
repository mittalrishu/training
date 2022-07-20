#code for Logistic Regression


import os 

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn import metrics
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import  PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.tsa as tsa


data=pd.read_csv("creditcard.csv")


data.head()

data.info()

fraud = data[data['Class'] == 1]
legit = data[data['Class'] == 0]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

X = data.drop('Class', axis = 1).values
y = data['Class'].values

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

np.bincount(y_train)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 1)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

n_errors = (y_pred != y_test).sum()
print("The model used is Logistic Regression")
  
acc = accuracy_score(y_test, y_pred)
print("The accuracy is {}".format(acc))
  
cr = classification_report(y_test, y_pred)
print("The classification report is{}".format(cr))

LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()