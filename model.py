# importing required libraries:
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from math import sqrt

# importing datasets:
data=pd.read_csv('final_dataset (1).csv')
data.head()

# dropping unnecessary column:
data.drop('Unnamed: 0',axis=1,inplace=True)

# Splitting into independent and dependent variable
X=data.drop("Top100",axis=1)    # excluding output column
y=data["Top100"]

# Applying SMOTE: to balance the dataset
from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy="minority")
X_sm,y_sm=smote.fit_resample(X,y)
print(X_sm.shape,y_sm.shape)


column=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
       'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Genre']
X_balance=pd.DataFrame(X_sm,columns=column)
X_balance.head(2)


# MinMax scaler to scale our data , since we can see that there can be units difference
scaling=MinMaxScaler()
X_scaled=scaling.fit_transform(X_sm)


column=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
       'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Genre']
X_final=pd.DataFrame(X_scaled,columns=column)
X_final.head(2)

# splitting the dataset into train and test:
X_train, X_test, y_train, y_test = train_test_split(X_final, y_sm, train_size = 0.8, test_size=0.2, random_state=15)

#calling the logistic model and fitting with best parameter:
log_reg2=LogisticRegression(C=1.623776739188721,penalty='l2',max_iter=100,solver='lbfgs')

# fitting the data for training:
log_reg2.fit(X_train,y_train)

#saving the model:
import pickle
pickle.dump(log_reg2,open("log_dep.pkl","wb"))

#loading model to compare the result:
model=pickle.load(open("log_dep.pkl","rb"))

#checking the prediction
print(model.predict([[1,5,6,7,8,6,6,8,15,20]]))



