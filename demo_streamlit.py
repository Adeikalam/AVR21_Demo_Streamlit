# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:00:01 2021

@author: Pierre
"""

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split


st.title("Voici mon projet de Data Science")


df = pd.read_csv("train.csv", index_col = 'PassengerId')

# Data Cleaning
df = df.drop(['Name', 'Cabin', 'Ticket'], axis = 1 )

df['Age'] = df['Age'].fillna(df['Age'].mode()[0])

df = pd.get_dummies(df)

st.write(df)


X = df.drop('Survived', axis = 1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)









