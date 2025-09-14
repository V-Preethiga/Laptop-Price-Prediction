# Laptop-Price-Prediction
This project predicts the rating of laptops based on their specifications such as brand, processor type, RAM, storage, GPU, operating system, weight, and other features. By analyzing these specifications, the model classifies laptops into rating categories.

import pandas as pd
from sklearn import preprocessing
data=pd.read_csv("laptopPrice.csv")
print(data.head(6))
data.isna().sum()
data.isnull().sum()
features=data.drop("rating",axis=1)
target=data["rating"]
label_encoder=preprocessing.LabelEncoder()
features['brand']=label_encoder.fit_transform(features['brand'])
features['brand'].unique()
features['processor_brand']=label_encoder.fit_transform(features['processor_brand'])
features['processor_brand'].unique()


features['processor_name']=label_encoder.fit_transform(features['processor_name'])
features['processor_name'].unique()


features['processor_gnrtn']=label_encoder.fit_transform(features['processor_gnrtn'])
features['processor_gnrtn'].unique()

features['ram_gb']=label_encoder.fit_transform(features['ram_gb'])
features['ram_gb'].unique()

features['ram_type']=label_encoder.fit_transform(features['ram_type'])
features['ram_type'].unique()

features['ssd']=label_encoder.fit_transform(features['ssd'])
features['ssd'].unique()

features['hdd']=label_encoder.fit_transform(features['hdd'])
features['hdd'].unique()

features['os']=label_encoder.fit_transform(features['os'])
features['os'].unique()



features['os_bit']=label_encoder.fit_transform(features['os_bit'])
features['os_bit'].unique() 

features['graphic_card_gb']=label_encoder.fit_transform(features['graphic_card_gb'])
features['graphic_card_gb'].unique()

features['weight']=label_encoder.fit_transform(features['weight'])
features['weight'].unique()

features['warranty']=label_encoder.fit_transform(features['warranty'])
features['warranty'].unique()

features['Touchscreen']=label_encoder.fit_transform(features['Touchscreen'])
features['Touchscreen'].unique()

features['msoffice']=label_encoder.fit_transform(features['msoffice'])
features['msoffice'].unique()

data['rating']=label_encoder.fit_transform(data['rating'])
data['rating'].unique()
target=data['rating']

import numpy as np
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=42)
rf_1=rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
accuracy= accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_pred,y_test)
plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True)
plt.show()
