# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:58:28 2020

@author: Mayank
"""
# Predicting the income

#To work with dataframes
import pandas as pd

#To perform numerical operations
import numpy as np

#To visualize data/graphs
import seaborn as sns

#To partition the data
from sklearn.model_selection import train_test_split

#Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

#Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

#Importing data
data_income = pd.read_csv('C:/project/adult.csv')

#Creating a copy of original data
data = data_income.copy()

#Data preprocessing

#To check variables data type
print(data.info())

#Checking missing values
data.isnull()

print('Data columns with null values:\n',data.isnull().sum())

#Summary of numerical variables
summary_num = data.describe()
print(summary_num)

#Summary of categorical variables
summary_cat = data.describe(include="O")
print(summary_cat)

#Frequency of each categories                            
print(data['workclass'].value_counts())
print(data['occupation'].value_counts())
print(data['capital.gain'].value_counts())
print(data['capital.loss'].value_counts())
print(data['race'].value_counts())
print(data['native.country'].value_counts())

#Checking for unique classes
print(np.unique(data['workclass']))
print(np.unique(data['occupation']))

#Consider '?' as na values
data = pd.read_csv('C:/project/adult.csv',na_values=["?"])

#Checking missing values
data.isnull().sum()

#Deleting rows with missing values
missing = data[data.isnull().any(axis=1)]

data2 = data.dropna(axis=0)

data2.isnull().sum()

#Data visualization/Exploratory analysis

#Relationship between independent variables
correlation = data2.corr()

#Extracting the column names
data2.columns

#Numerical Data Analysis
num_attributes = data2.select_dtypes(include=['int64'])
print(num_attributes.columns)
num_attributes.hist(figsize=(10,10))

cat_attributes = data2.select_dtypes(include=['object'])
print(cat_attributes.columns)

#Gender vs Income Analysis
gender = pd.crosstab( index = data2["sex"], columns = 'count', normalize = True)
gender_income = pd.crosstab( index = data2["sex"],columns = data2["income"], normalize = 'index', margins = True)
sns.countplot(y='sex', hue='income', data = cat_attributes)

#Age vs Income Analysis
age = pd.crosstab( index = data2["age"], columns = 'count', normalize = True)
data2.groupby('income')["age"].median()
sns.distplot(data2["age"], bins=10,kde= False)
sns.boxplot("income","age",data=data2)

#Workclass vs Income Analysis
workclass_income = pd.crosstab( index = data2["workclass"], columns = data2["income"], normalize = 'index', margins = True)
sns.countplot(y='workclass', hue='income', data = cat_attributes)

#Education vs Income Analysis
education_income = pd.crosstab( index = data2["education"], columns = data2["income"], normalize = 'index', margins = True)
sns.countplot(y='education', hue='income', data = cat_attributes)

#Race vs Income Analysis
race_income = pd.crosstab( index = data2["race"], columns = data2["income"], normalize = 'index', margins = True)
sns.countplot(y='race', hue='income', data = cat_attributes)

#Native Country vs Income Analysis
nativecountry_income = pd.crosstab( index = data2["native.country"], columns = data2["income"], normalize = 'index', margins = True)
sns.countplot(y='native.country', hue='income', data = cat_attributes)

#Marital Status vs Income Analysis
maritalstatus_income = pd.crosstab( index = data2["marital.status"], columns = data2["income"], normalize = 'index', margins = True)
sns.countplot(y='marital.status', hue='income', data = cat_attributes)

#Capitalgain vs Income Analysis
capitalgain = pd.crosstab( index = data2["capital.gain"], columns = 'count', normalize = True)
sns.distplot(data2["capital.gain"], bins=10,kde= False)
sns.boxplot("income","capital.gain",data=data2)

#Capitalloss vs Income Analysis
capitalloss = pd.crosstab( index = data2["capital.loss"], columns = 'count', normalize = True)
sns.distplot(data2["capital.loss"], bins=10,kde= False)
sns.boxplot("income","capital.loss",data=data2)

#Hoursperweek vs Income Analysis
hoursperweek = pd.crosstab( index = data2["hours.per.week"], columns = 'count', normalize = True)
data2.groupby('income')["hours.per.week"].median()
sns.distplot(data2["hours.per.week"], bins=10,kde= False)
sns.boxplot("income","hours.per.week",data=data2)

#Logistic Regression

#Reindexing the income to 0,1
data2['income']=data2['income'].map({'<=50K':0,'>50K':1})
print(data2['income'])

new_data=pd.get_dummies(data2, drop_first=True)

#Storing the column names
columns_list=list(new_data.columns)
print(columns_list)

#Separating the input names from data
features=list(set(columns_list)-set(['income']))
print(features)

#Storing the output values in y
y=new_data['income'].values
print(y)

#Storing the values from input features
x=new_data[features].values
print(x)

#Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the vakues for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# Prediction from test data
prediction1 = logistic.predict(test_x)
print(prediction1)

# Confusion matrix3
confusion_matrix1= confusion_matrix(test_y,prediction1)
print(confusion_matrix1)

# Accuracy calculation
accuracy_score1 = accuracy_score(test_y,prediction1)
print(accuracy_score1)

# Missclassified values from prediction
print("Missclassified samples : %d" % (test_y != prediction1).sum())

# KNN Classifier

# importing library for KNN
from sklearn.neighbors import KNeighborsClassifier

# Storing the K nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 18)

# Fitting the values for x and y
KNN_classifier.fit(train_x,train_y)

# Predicting the test values with model
prediction2 = KNN_classifier.predict(test_x)

# Performance metric check
confusion_matrix2 = confusion_matrix(test_y,prediction2)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix2)

# Accuracy calculation
accuracy_score2 = accuracy_score(test_y,prediction2)
print(accuracy_score2)

# Missclassified values from prediction
print("Missclassified samples : %d" % (test_y != prediction2).sum())

# Calculating error for K values between 1 and 20
Misclassified_sample=[]
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())
    
print(Misclassified_sample)

# Removing insignificant variables

#Reindexing the income to 0,1
data2['income']=data2['income'].map({'<=50K':0,'>50K':1})
print(data2['income'])

# Education vs Education.num
education_merge = pd.crosstab( index = data2["education"],columns = data2["education.num"], normalize = 'index', margins = True)

#Race vs Income Analysis
race_income = pd.crosstab( index = data2["race"], columns = data2["income"], normalize = 'index', margins = True)
sns.countplot(y='race', hue='income', data = cat_attributes)
print(data['race'].value_counts())

#Native Country vs Income Analysis
nativecountry_income = pd.crosstab( index = data2["native.country"], columns = data2["income"], normalize = 'index', margins = True)
sns.countplot(y='native.country', hue='income', data = cat_attributes)
print(data['native.country'].value_counts())

cols = ['education','native.country','race',]
new_data = data2.drop(cols,axis=1)

new_data=pd.get_dummies(new_data, drop_first=True)

#Storing the column names
columns_list=list(new_data.columns)
print(columns_list)

#Separating the input names from data
features=list(set(columns_list)-set(['income']))
print(features)

#Storing the output values in y
y=new_data['income'].values
print(y)

#Storing the values from input features
x=new_data[features].values
print(x)

#Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=1)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the vakues for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# Prediction from test data
prediction1 = logistic.predict(test_x)
print(prediction1)

# Confusion matrix
confusion_matrix1= confusion_matrix(test_y,prediction1)
print(confusion_matrix1)

# Accuracy calculation
accuracy_score1 = accuracy_score(test_y,prediction1)
print(accuracy_score1)

# Random Forest

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10)

# Train the model on training data
rf.fit(train_x, train_y)

# Use the forest's predict method on the test data
prediction3 = rf.predict(test_x)
print(prediction3)
prediction3= np.around(prediction3)

prediction3 = prediction3.astype(int)
print(prediction3)

# Confusion matrix
confusion_matrix3= confusion_matrix(test_y,prediction3)
print(confusion_matrix3)

# Accuracy calculation
accuracy_score3 = accuracy_score(test_y,prediction3)
print(accuracy_score3)

# Calculate the absolute errors
errors = abs(prediction3 - test_y)
print(errors)

# Missclassified values from prediction
print("Missclassified samples : %d" % (test_y != prediction3).sum())

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print(test_y)
data_accuracy = pd.read_csv('C:/project/accuracy.csv')
from bubble_plot.bubble_plot import bubble_plot
bubble_plot(data_accuracy,'test_y','prediction1', normalization_by_all=False)
data_accuracy2 = pd.read_csv('C:/project/accuracy2.csv')
bubble_plot(data_accuracy2,'test_y','prediction2', normalization_by_all=False)
data_accuracy3 = pd.read_csv('C:/project/accuracy3.csv')
bubble_plot(data_accuracy3,'test_y','prediction3', normalization_by_all=False)