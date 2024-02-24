#import library 
import pandas as pd
import numpy as np
training = pd.read_csv('D:\\Studying\\Visual Studio 2022\\DiseasePrediction\\archive\\Training.csv')
training['prognosis'].unique()

# import label encoder 
from sklearn import preprocessing 
label_encoder=preprocessing.LabelEncoder()
# Encode label in 'prognosis' column
training['prognosis']=label_encoder.fit_transform(training['prognosis'])
training['prognosis'].unique()
del training['Unnamed: 133']
"""Checking whether data is balanced or not"""
# import matplotlib.pyplot as plt 
# import seaborn as sns
# disease_counts = training["prognosis"].value_counts()
# temp_df = pd.DataFrame({
#     "Disease": disease_counts.index,
#     "Counts": disease_counts.values
# })
 
# plt.figure(figsize = (16,0))
# sns.barplot(x = "Disease", y = "Counts", data = temp_df)
# plt.xticks(rotation=90)
# plt.show()
""" Performing k-fold validation to X_train for model selection
     split training into train_data and test_data """
from sklearn.model_selection import train_test_split
X = training.iloc[:,:-1]
y = training.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 24)
