#!/usr/bin/env python
# coding: utf-8

# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# 
# model = joblib.load('titanic_model.pkl')
# scaler = joblib.load('titanic_scaler.pkl')
# 
# # Streamlit App
# st.title('Titanic Survival Prediction App')
# st.write('''This app predicts the survival of passengers on the Titanic based on the features provided in the dataset.''')  
# # Pclass - 1,2,3
# Pclass = st.selectbox('Select the Ticket Class', [1, 2, 3])
# # Age - 0 to 100
# Age = st.slider('Select the Age of the Passenger', 0, 100)
# # Fare - 0 to 1000
# Fare = st.slider('Select the Fare of the Passenger', 0, 1000)
# # SibSp - 0,1,2,3,4,5,8
# SibSp = st.selectbox('Select the Number of Sibling and Spouse', [0, 1, 2, 3, 4, 5, 8])
# # Parch - 0,1,2,3,4,5,6
# Parch = st.selectbox('Select the Number of Parents and Children', [0, 1, 2, 3, 4, 5, 6])
# # Embarked - C,Q,S
# Embarked = st.selectbox('Select the Port of Embarkation', ['C', 'Q', 'S'])
# 
# # FamilySize = SibSp + Parch + 1
# Family_Size = SibSp + Parch + 1
# # isAlone = 1 if FamilySize == 1 else 0
# isAlone = 1 if Family_Size == 1 else 0
# 
# # Title - Mr, Miss, Mrs, Master
# Title = st.selectbox('Select the Title of the Passenger', ['Mr', 'Miss', 'Mrs', 'Master'])
# # GenderClass
# GenderClass = st.selectbox('Select the GenderClass of the Passenger', ['Male', 'Female', 'child'])
# 
# # Create a DataFrame
# data = {
#     'Pclass': Pclass,
#     'Age': Age,
#     'Fare': Fare,
#     'Family_Size': Family_Size,
#     'isAlone': isAlone,
#     'Embarked': Embarked,
#     'GenderClass': GenderClass,
#     'Title': Title
# }
# 
# df = pd.DataFrame(data, index=[0])
# print("BYEEEEEEE")
# 
# # Dummification
# # columns_to_dummify = ['Embarked', 'Title', 'GenderClass']
# 
# 
# if df['Embarked'][0] == 'C':
#     df['Embarked_C'] = 1
#     df['Embarked_Q'] = 0
#     df['Embarked_S'] = 0
# elif df['Embarked'][0] == 'Q':
#     df['Embarked_C'] = 0
#     df['Embarked_Q'] = 1
#     df['Embarked_S'] = 0
# else:
#     df['Embarked_C'] = 0
#     df['Embarked_Q'] = 0
#     df['Embarked_S'] = 1
# if df['Title'][0] == 'Miss':
#     df['Title_Miss'] = 1
#     df['Title_Mr'] = 0
#     df['Title_Mrs'] = 0
#     df['Title_Master'] = 0
# elif df['Title'][0] == 'Mr':
#     df['Title_Miss'] = 0
#     df['Title_Mr'] = 1
#     df['Title_Mrs'] = 0
#     df['Title_Master'] = 0
# elif df['Title'][0] == 'Mrs':
#     df['Title_Miss'] = 0
#     df['Title_Mr'] = 0
#     df['Title_Mrs'] = 1
#     df['Title_Master'] = 0
# else:
#     df['Title_Miss'] = 0
#     df['Title_Mr'] = 0
#     df['Title_Mrs'] = 0
#     df['Title_Master'] = 1
# if df['GenderClass'][0] == 'Male':
#     df['GenderClass_child'] = 0
#     df['GenderClass_male'] = 1
#     df['GenderClass_female'] = 0
# elif df['GenderClass'][0] == 'Female':
#     df['GenderClass_child'] = 0
#     df['GenderClass_male'] = 0
#     df['GenderClass_female'] = 1
# else:
#     df['GenderClass_child'] = 1
#     df['GenderClass_male'] = 0
#     df['GenderClass_femle'] = 0
# df.drop(['Embarked', 'Title', "GenderClass"], axis=1, inplace=True)
# 
# # Modify the DataFrame
# df = df[['Pclass', 'Age', 'Fare', 'Family_Size', 'isAlone', 'Embarked_Q',
#        'Embarked_S', 'GenderClass_female', 'GenderClass_male', 'Title_Miss',
#        'Title_Mr', 'Title_Mrs']]
# 
# # Scaling
# columns_to_be_scaled = ['Age', 'Fare']
# df[columns_to_be_scaled] = scaler.transform(df[columns_to_be_scaled])
# # Display the DataFrame
# print(df)
# 
# # Prediction
# prediction = model.predict(df)
# if prediction[0] == 0:
#     st.error("The passenger did not survive.")
# else:
#     st.success("The passenger survived.")
# st.write(prediction[0])

# In[ ]:


# Deployment of the Model - Streamlit
# Deployment is the process of making the model available for the end-users.
# Streamlit is an open-source Python library that makes it easy to create and share web apps for 
# machine learning and data science.

# Installation of Streamlit
# !pip install streamlit -q

# Create a new Python file called app.py
# Open the file in any IDE and write the following code
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the model and the scaler
model = joblib.load('titanic_model.pkl')
scaler = joblib.load('titanic_scaler.pkl')
# Streamlit App
st.title('Titanic Survival Prediction App')
st.write('''This app predicts the survival of passengers on the Titanic based on the features provided in the dataset.''')  
# Pclass - 1,2,3
pclass = st.selectbox('Select the Ticket Class', [1, 2, 3])
# Age - 0 to 100
age = st.slider('Select the Age of the Passenger', 0, 100)
# Fare - 0 to 1000
fare = st.slider('Select the Fare of the Passenger', 0, 1000)
# SibSp - 0,1,2,3,4,5,8
sibsp = st.selectbox('Select the Number of Sibling and Spouse', [0, 1, 2, 3, 4, 5, 8])
# Parch - 0,1,2,3,4,5,6
parch = st.selectbox('Select the Number of Parents and Children', [0, 1, 2, 3, 4, 5, 6])
# Embarked - C,Q,S
embarked = st.selectbox('Select the Port of Embarkation', ['C', 'Q', 'S'])
# FamilySize = SibSp + Parch + 1
family_size = sibsp + parch + 1
# isAlone = 1 if FamilySize == 1 else 0
is_alone = 1 if family_size == 1 else 0
# Title - Mr, Miss, Mrs, Master
title = st.selectbox('Select the Title of the Passenger', ['Mr', 'Miss', 'Mrs', 'Master'])
# GenderClass
gender_class = st.selectbox('Select the GenderClass of the Passenger', ['Male', 'Female', 'child'])
# Final column list: 
# ['Pclass', 'Age', 'Fare', 'FamilySize', 'isAlone', 'Embarked', 'Title', 'GenderClass']

# ['Pclass', 'Age', 'Fare', 'FamilySize', 'isAlone', 'Embarked_Q',
#        'Embarked_S', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
#        'GenderClass_female', 'GenderClass_male']
# Dummification on Embarked, Title, GenderClass

# Create a DataFrame
data = {
    'Pclass': pclass,
    'Age': age,
    'Fare': fare,
    'FamilySize': family_size,
    'isAlone': is_alone,
    'Embarked': embarked,
    'Title': title,
    'GenderClass': gender_class
}
df = pd.DataFrame(data, index=[0])
# print("HIIIII")
# Display the DataFrame
# print(df)
# print(df.dtypes)
# Dummification
# columns_to_dummify = ['Embarked', 'Title', 'GenderClass']
# df = pd.get_dummies(df, columns=['Embarked', 'Title', 'GenderClass'], dtype="int")
print("BYEEEEEEE")
# Display the DataFrame
# print(df.dtypes)
if df['Embarked'][0] == 'C':
    df['Embarked_C'] = 1
    df['Embarked_Q'] = 0
    df['Embarked_S'] = 0
elif df['Embarked'][0] == 'Q':
    df['Embarked_C'] = 0
    df['Embarked_Q'] = 1
    df['Embarked_S'] = 0
else:
    df['Embarked_C'] = 0
    df['Embarked_Q'] = 0
    df['Embarked_S'] = 1
if df['Title'][0] == 'Miss':
    df['Title_Miss'] = 1
    df['Title_Mr'] = 0
    df['Title_Mrs'] = 0
    df['Title_Master'] = 0
elif df['Title'][0] == 'Mr':
    df['Title_Miss'] = 0
    df['Title_Mr'] = 1
    df['Title_Mrs'] = 0
    df['Title_Master'] = 0
elif df['Title'][0] == 'Mrs':
    df['Title_Miss'] = 0
    df['Title_Mr'] = 0
    df['Title_Mrs'] = 1
    df['Title_Master'] = 0
else:
    df['Title_Miss'] = 0
    df['Title_Mr'] = 0
    df['Title_Mrs'] = 0
    df['Title_Master'] = 1
if df['GenderClass'][0] == 'Male':
    df['GenderClass_child'] = 0
    df['GenderClass_male'] = 1
    df['GenderClass_female'] = 0
elif df['GenderClass'][0] == 'Female':
    df['GenderClass_child'] = 0
    df['GenderClass_male'] = 0
    df['GenderClass_female'] = 1
else:
    df['GenderClass_child'] = 1
    df['GenderClass_male'] = 0
    df['GenderClass_female'] = 0
df.drop(['Embarked', 'Title', "GenderClass"], axis=1, inplace=True)
# Modify the DataFrame
# ['Pclass', 'Age', 'Fare', 'FamilySize', 'isAlone', 'Embarked_Q',
#        'Embarked_S', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
#        'GenderClass_female', 'GenderClass_male']
df = df[['Pclass', 'Age', 'Fare', 'FamilySize', 'isAlone', 'Embarked_Q', 
         'Embarked_S', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
       'GenderClass_female', 'GenderClass_male']]

# print(df)
# Scaling
columns_to_be_scaled = ['Age', 'Fare']
df[columns_to_be_scaled] = scaler.transform(df[columns_to_be_scaled])
# Display the DataFrame
print(df)
# Prediction
prediction = model.predict(df)
if prediction[0] == 0:
    st.error("The passenger did not survive.")
else:
    st.success("The passenger survived.")
st.write(prediction[0])

