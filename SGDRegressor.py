# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:28:13 2018

@author: Sumant Kulkarni
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler

# Set a random seed
import random
random.seed(42)

# Load the dataset
in_file = 'Strategic_Subject_List_Clean.csv'
full_data = pd.read_csv(in_file, low_memory=False,
                        dtype = {
'SSL SCORE': np.float32,
'PREDICTOR RAT AGE AT LATEST ARREST': str,
'PREDICTOR RAT VICTIM SHOOTING INCIDENTS': np.int32,
'PREDICTOR RAT VICTIM BATTERY OR ASSAULT': np.int32,
'PREDICTOR RAT ARRESTS VIOLENT OFFENSES': np.int32,
'PREDICTOR RAT GANG AFFILIATION': np.int32,
'PREDICTOR RAT NARCOTIC ARRESTS': np.int32,
'PREDICTOR RAT TREND IN CRIMINAL ACTIVITY': np.float32,
'PREDICTOR RAT UUW ARRESTS': np.int32,
'SEX CODE CD': str,
'RACE CODE CD': str,
'WEAPON I': str,
'DRUG I': str,
'AGE GROUP': str,
'STOP ORDER NO': str,
'PAROLEE I': str,
'LATEST DATE': str,
'LATEST DIST': str,
'MAJORITY DIST': str,
'DLST': str,
'LATEST DIST RES': str,
'WEAPONS ARR CNT': np.float32,
'LATEST WEAPON ARR DATE': str,
'NARCOTICS ARR CNT': np.float32,
'LATEST NARCOTIC ARR DATE': str,
'IDOC RES CITY': str,
'IDOC RES STATE CODE': str,
'IDOC CPD DIST': str,
'CPD ARREST I': str,
'DOMESTIC ARR CNT': np.float32,
'LATEST DOMESTIC ARR DATE': str,
'AGE CURR': str,
'SSL LAST PTV DATE': str,
'TRAP STATUS': np.float32,
'TRAP FLAGS': np.float32,
'SSL FLAGS': np.float32,
'COMMUNITY AREA': str
})

pd.isnull(full_data).sum() > 0
full_data['PREDICTOR RAT AGE AT LATEST ARREST']=full_data['PREDICTOR RAT AGE AT LATEST ARREST'].fillna("")
full_data['AGE GROUP']=full_data['AGE GROUP'].fillna("")
full_data['STOP ORDER NO']=full_data['STOP ORDER NO'].fillna("")
full_data['PAROLEE I']=full_data['PAROLEE I'].fillna("")
full_data['MAJORITY DIST']=full_data['MAJORITY DIST'].fillna("")
full_data['DLST']=full_data['DLST'].fillna("")
full_data['WEAPONS ARR CNT']=full_data['WEAPONS ARR CNT'].fillna(0.0)
full_data['LATEST WEAPON ARR DATE']=full_data['LATEST WEAPON ARR DATE'].fillna("")
full_data['NARCOTICS ARR CNT']=full_data['NARCOTICS ARR CNT'].fillna(0.0)
full_data['LATEST NARCOTIC ARR DATE']=full_data['LATEST NARCOTIC ARR DATE'].fillna("")
full_data['IDOC RES CITY']=full_data['IDOC RES CITY'].fillna("")
full_data['IDOC RES STATE CODE']=full_data['IDOC RES STATE CODE'].fillna("")
full_data['IDOC CPD DIST']=full_data['IDOC CPD DIST'].fillna("")
full_data['DOMESTIC ARR CNT']=full_data['DOMESTIC ARR CNT'].fillna(0.0)
full_data['LATEST DOMESTIC ARR DATE']=full_data['LATEST DOMESTIC ARR DATE'].fillna("")
full_data['AGE CURR']=full_data['AGE CURR'].fillna("")
full_data['SSL LAST PTV DATE']=full_data['SSL LAST PTV DATE'].fillna("")
full_data['TRAP STATUS']=full_data['TRAP STATUS'].fillna(0.0)
full_data['TRAP FLAGS']=full_data['TRAP FLAGS'].fillna(0.0)
full_data['SSL FLAGS']=full_data['SSL FLAGS'].fillna(0.0)


#parse_dates=parse_dates = ['col1', 'col2']

# Print the first few entries of the dataset
display(full_data.head())

# Store the 'SSL Score' feature in a new variable and remove it from the dataset
outcomes = full_data['SSL SCORE']
features_raw = full_data.drop('SSL SCORE', axis = 1)

# Show the new dataset with 'SSL SCORE' removed
display(features_raw.head())

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['PREDICTOR RAT VICTIM SHOOTING INCIDENTS', 
             'PREDICTOR RAT VICTIM BATTERY OR ASSAULT', 
             'PREDICTOR RAT ARRESTS VIOLENT OFFENSES', 
             'PREDICTOR RAT GANG AFFILIATION', 
             'PREDICTOR RAT NARCOTIC ARRESTS', 
             'PREDICTOR RAT TREND IN CRIMINAL ACTIVITY',
             'PREDICTOR RAT UUW ARRESTS',
             'WEAPONS ARR CNT',
             'NARCOTICS ARR CNT',
             'DOMESTIC ARR CNT',
             'TRAP STATUS',
             'TRAP FLAGS',
             'SSL FLAGS']

features_minmax_transform = pd.DataFrame(data = features_raw)
features_minmax_transform[numerical] = scaler.fit_transform(features_raw[numerical])

# Show an example of a record with scaling applied
display(features_minmax_transform.head(n = 5))

#One hot encoding
features_final =  pd.get_dummies(features_minmax_transform)

# Split the 'features' into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    outcomes, 
                                                    test_size = 0.7, 
                                                    random_state = 0)


# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# Training the model
clf = linear_model.SGDRegressor()
clf.fit(X_train, y_train)

# Making predictions
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculating accuracies
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)

