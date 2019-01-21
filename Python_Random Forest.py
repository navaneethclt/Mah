
"""
RANDOM FOREST
"""

##############    Random forest #########
################# required module######################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# importing library for getting parameters
from pprint import pprint

# importing the library for randomforest 
from sklearn.ensemble import RandomForestRegressor

#from sklearn.tree import export_graphviz
#from pprint import pprint

# importing the library for randomizedsearchCV
from sklearn.model_selection import RandomizedSearchCV

# read data set
data=pd.read_csv('ToyotaCorollaNew.csv')

#"""
#Exploratory data analysis:

#1.Data information (types and missing value) and statistics
#2.Data preprocessing (Missing value,handling categorical variables etc)
#3.Data visualization

print(data.info()) ###get data information

print('Data columns with null values:\n', data.isnull().sum()) ##missing value

data['Age'].fillna(data['Age'].mean(), inplace = True)
data['FuelType'].fillna(data['FuelType'].mode()[0], inplace = True)
data['MetColor'].fillna(data['MetColor'].mode()[0], inplace = True)

print('Data columns with null values:\n', data.isnull().sum()) ##missing value
data.drop('Unnamed: 0',axis=1,inplace=True)

# =============================================================================

# =============================================================================
data=pd.get_dummies(data)##convert string into dummy variable
data = data.astype(float) ##convert all integer into float
features=list(set(data.columns)-set(['Price']))
target=list(['Price'])

###################################################
# Separating out the features
x = data.loc[:, features].values
y = data.loc[:,target].values
train_x, test_x, train_y, test_y = train_test_split( x, y, test_size=0.3, random_state=1)

##################base line model####################################

base_pred=data[target].mean().values
# 
base_pred=np.repeat(base_pred, len(test_y))

base_root_mean_square_error=(mean_squared_error(test_y, base_pred))**0.5



############################################################

scaler = StandardScaler()# Fit on training set only.
scaler.fit(train_x)
# Apply transform to both the training set and the test set.
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

###########################Random forest#####################################

# Building a Random forest model

rf = RandomForestRegressor(n_estimators = 100,random_state=1)

rf.fit(train_x, train_y.ravel());

# Use the forest's predict method on the test data
predictions_rf = rf.predict(test_x)

# Calculate the absolute errors
rf_root_mean_square_error=(mean_squared_error(test_y, predictions_rf))**0.5
print(rf_root_mean_square_error)

#
#

"""
Title: Random forest with important features 
In this task we will use subset of features and compare results with the 
base line and the above randome forest model
"""
importances = list(rf.feature_importances_)
#
## List of tuples with variable and importance
feature_importances = [(features, round(importance, 4)) for features, 
                       importance in zip(features, importances)]

print(feature_importances)
#
## Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], 
                                                         reverse = True)
print(feature_importances)

#
[print('Variable: {:20} Importance: {}'.format(*pair))for pair in feature_importances];
#
#
rf_most_important = RandomForestRegressor(n_estimators=100, random_state=1)
#

important_features_positions=[]

##  Select top 5 important features
for i in range(5):
    important_features=feature_importances[i][0]
    index=features.index(important_features)
    important_features_positions.append(index)
##selecting important features from train data    
train_x_important=train_x[:,important_features_positions] 
##selecting important features from train data
test_x_important=test_x[:,important_features_positions] 
rf_most_important.fit(train_x_important, train_y.ravel());
predictions_rf_most_important = rf_most_important.predict(test_x_important)

##
##

## Calculate the absolute errors
rf_root_mean_square_error_most_imp=(mean_squared_error(test_y, 
                                predictions_rf_most_important))**0.5
print(rf_root_mean_square_error_most_imp)

#

"""Hyper parameter tuning"""

## Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

## Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 400,num = 10)]
print(n_estimators)

## Number of features to consider at every split
max_features = ['auto', 'sqrt']

## Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

## Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

## Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

## Method of selecting samples for training each tree
bootstrap = [True, False]

## Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

## Use the random grid to search for best hyperparameters
## First create the base model to tune
rf_for_tuning = RandomForestRegressor()

## Random search of parameters, using 3 fold cross validation, 
## search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf_for_tuning, 
                            param_distributions = random_grid, 
                            n_iter = 100, cv = 3, verbose=2, random_state=1)


## Fit the random search model
rf_random.fit(train_x, train_y.ravel())
print(rf_random.best_params_)

## finding the best model
rf_model_best = rf_random.best_estimator_
print(rf_model_best)

# predicting with the test data on best model
predictions_best = rf_model_best.predict(test_x)

rf_root_mean_square_error_best=(mean_squared_error(test_y,
                                        predictions_best))**0.5
                                                  
print(rf_root_mean_square_error_best)
