# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:14:51 2019

@author: SENAIIITM
"""

import pandas as pd
import itertools
from pandas import ExcelWriter
from pandas import ExcelFile
import xlsxwriter
import matplotlib as plt
import datetime
import numpy as np
import matplotlib.cm as cm
from scipy.interpolate import interp1d
import math 



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score , confusion_matrix , precision_score , recall_score 
from sklearn.impute import SimpleImputer 
# importing library for getting parameters
from pprint import pprint

# importing the library for randomforest 
from sklearn.ensemble import RandomForestClassifier

#from sklearn.tree import export_graphviz
#from pprint import pprint

# importing the library for randomizedsearchCV
from sklearn.model_selection import RandomizedSearchCV

xls = pd.ExcelFile('May.xlsx',index = False);
#%%

AprilData = xls.parse('Sheet1', index =False, na_values=['NA']);
#%%
AprilData = AprilData.reindex(index = AprilData.index[::-1])
AprilData.reset_index(drop=True, inplace=True)
AprilData = AprilData.dropna(axis = 1);
AprilData = AprilData[AprilData['Parameter Value']< 30000]
AprilData = AprilData[AprilData['Machine Parameter']!= 'SPRAY_PUMP_MOTOR__A_TEMPERATURE']
AprilData['Machine Parameter'] = AprilData['Machine Parameter'].str.replace(" ","_")
SensorNames = AprilData.drop_duplicates('Machine Parameter');

#sdf = AprilData.groupby(['Machine Parameter'],group_keys = True)
#params = SensorsNames.groupby(['Machine Parameter']).groups.keys();
#AprilData[AprilData['Machine Parameter'] == 'TROLLY MOTOR CURRENT']
params = SensorNames.groupby(['Machine Parameter']).groups.keys();
AprilData['Inserted Date'] = AprilData['Inserted Date'].dt.round('1s')
#AprilData[AprilData['Machine Parameter'] == 'TROLLY MOTOR CURRENT']

companies = list(params)
i = 1
for c in companies:
         a = 'df' + str(i)
         b = 'temp'+str(i)
    
         exec(a + "= AprilData[AprilData['Machine Parameter'] == c]")
         exec(a + " = "+a+".groupby('Inserted Date',as_index =True)['Parameter Value'].mean()")
         #exec(b+"="+a+".resample('7s',on='Inserted Date').last()")
        
#forward filling is questionable since we dont want data preceding a failure to be propagated throughout the dataset
         #exec(b+".fillna(method='ffill',inplace=True)") 
         i = i + 1
timeindex = AprilData['Inserted Date'].drop_duplicates()
timeindex.sort_values(ascending=True, kind='quicksort', na_position='last', inplace=True)
working_time = []

for date in timeindex:
    if date.hour>=8 and (date.hour<=16 and date.minute<=30):
        if not(date.hour==11 and date.minute>=30):
            if not(date.hour==15 and date.minute<=7):
               working_time.append(date)
AprilDataRounded = pd.DataFrame(index = working_time , columns = companies)
AprilDataRounded['Inserted_Date'] = working_time
AprilDataAll = pd.DataFrame(index = timeindex , columns = companies)
AprilDataAll['Inserted_Date'] = list(timeindex)
for i in range(len(companies)):
    b = 'df' + str(i+1)
    sensor = companies[i]
    exec("AprilDataRounded['"+sensor+"']="+b+"")
    exec("AprilDataAll['"+sensor+"']="+b+"")
#%%
mxls = pd.ExcelFile('master.xlsx',index = False);
master = mxls.parse('Sheet1',index =False, na_values=['NA']);
#%%

#AprilDataCBM = pd.DataFrame();
#AprilDataCBMcols = {'Inserted Date','Machine Parameter','Parameter Value'}
#AprilDataCBM[list(AprilDataCBMcols)] = AprilData[list(AprilDataCBMcols)]
##AprilDataCBM['Current Value'] = AprilDataCBM['Parameter Value']/master.loc[master.Machine_Parameter == AprilDataCBM['Machine Parameter'],'Scale_Denominator'].iloc[0]
#for index,value in AprilDataCBM.iterrows():
#    AprilDataCBM.loc[index,'Min'] =  master.loc[master.Machine_Parameter==AprilDataCBM.loc[index,'Machine Parameter'],'Min_Val'].iloc[0]
#    AprilDataCBM.loc[index,'Max'] =  master.loc[master.Machine_Parameter==AprilDataCBM.loc[index,'Machine Parameter'],'Max_Val'].iloc[0]
#    AprilDataCBM.loc[index,'Scale'] =  master.loc[master.Machine_Parameter==AprilDataCBM.loc[index,'Machine Parameter'],'Scale_Denominator'].iloc[0]
#    print(index)
#%%

 i = 0
 freq12 = {}
 for index,value in df12.iterrows():
   if (index!=1):
    pres = df12.loc[index, 'Inserted Date']
    freq12[i] = (pres - prev).total_seconds()
    
    prev = pres
    i = i+1
   else : 
       prev = df12.loc[index, 'Inserted Date']
       i = i+1

#%%
#DataImpcols = AprilData['Machine Parameter'].drop_duplicates() 
#DataImp = pd.DataFrame(columns = DataImpcols)
#DataImp.insert(0,'Inserted_Date', AprilData['Inserted Date'])
#for value in DataImpcols:
#    DataImp[value] = AprilData['Parameter Value'].where(AprilData['Machine Parameter'] == value)
##both ffill and bfill are questionable since they will wrongly propagate error
##DataImp.fillna(method = 'ffill',inplace = True)
##DataImp.fillna(method = 'bfill',inplace = True)
#DataImp.drop(['SPRAY_PUMP_MOTOR__A_TEMPERATURE'],inplace = True,axis = 1)
#%%
xlsal = pd.ExcelFile('Crankcase Cleaning Machine- Alarm Data2018.xlsx',index = False);
xlsalcls = pd.ExcelFile('Alarmsignal_classification.xlsx',index = False);
#%%
AlarmData = xlsal.parse('Alarm data', index =False, na_values=['NA']);
AlarmClass = xlsalcls.parse('Fault Categories-CCCM (2)', index =False, na_values=['NA']);
AlarmClass.Priority = AlarmClass.Priority.str.capitalize()
AlarmNames = AlarmData.drop_duplicates('Alarm_Name')
AlarmGroup = AlarmData.groupby('Alarm_Name',group_keys = True)
AlarmClassList = AlarmClass.Alarm_Name
AlarmCount = AlarmData.groupby(['Alarm_Name']).size().reset_index(name='count')
AlarmPriorityGroup= AlarmData.groupby('Priority',group_keys = True)                                     
for index,value in AlarmData.iterrows():
    if AlarmData.loc[index,'Alarm_Name'] in set(AlarmClassList):
        AlarmData.loc[index,'Priority'] =  AlarmClass.loc[AlarmClass.Alarm_Name==AlarmData.loc[index,'Alarm_Name'],'Priority'].iloc[0]
for index,value in AlarmData.iterrows():
    if AlarmData.loc[index,'Alarm_Name'] in set(AlarmClassList):
        AlarmData.loc[index,'Alarm Source'] =  AlarmClass.loc[AlarmClass.Alarm_Name==AlarmData.loc[index,'Alarm_Name'],'Alarm Source'].iloc[0]

AlarmData['T/F'] = 'F'
for key,group in AlarmPriorityGroup:
    if(key == 'Minor stoppage'):
        mask = (group['Down_Time'] <= datetime.time(minute = 4)) & (group['Down_Time'] >= datetime.time(minute =1))
        mask = mask[mask == True]
        AlarmData.loc[mask.index,'T/F'] = 'T'
    if(key == 'Breakdown'):
        mask = (group['Down_Time'] <= datetime.time(minute = 20)) & (group['Down_Time'] >= datetime.time(minute = 1))
        mask = mask[mask == True]
        AlarmData.loc[mask.index,'T/F'] = 'T'
        
        
AlarmDataMachineGenerated = AlarmData[(AlarmData['Alarm Source']!='Triggered Manually') & (AlarmData['Priority']!='Chamber opened for repair/observation')]
AlarmDataMachineGeneratedTrueAlarms = AlarmDataMachineGenerated[AlarmDataMachineGenerated['T/F'] == 'T']
AlarmGroupMachineGenerated = AlarmDataMachineGenerated.groupby('Alarm_Name',group_keys = True)
AlarmDataMachineGenerated.fillna('NA', inplace = True)
AlarmNames = AlarmData.drop_duplicates('Alarm_Name')


categories = np.unique(AlarmDataMachineGenerated['Priority'])
colors = cm.rainbow(np.linspace(0, 1, len(categories)))
colordict = dict(zip(categories, colors))  
AlarmDataMachineGenerated["Color"] = AlarmDataMachineGenerated['Priority'].apply(lambda x: colordict[x])

#%%


#%%
 
for index,value in df1.iterrows():
 plt.pyplot.scatter(df1.loc[index,'Inserted Date'],df1.loc[index,'Parameter Value'],s = 1,color = 'red');
#%%
def subtime(x):
    return (x - datetime.datetime(2018,4,1)).total_seconds()
#%%
Timelist = []

AlarmDataTrueAlarms = AlarmData[AlarmData['T/F'] == 'T']

i = 0
for index,value in AlarmDataTrueAlarms.iterrows():
    Timelist.append((subtime(AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time']),subtime(AlarmDataTrueAlarms.loc[index,'Alarm_Stop_Time'])))
sorted_by_lower_bound = sorted(Timelist, key=lambda tup: tup[0])
merged = []

for higher in sorted_by_lower_bound:
    if not merged:
        merged.append(higher)
    else:
        lower = merged[-1]
        # test for intersection between lower and higher:
        # we know via sorting that lower[0] <= higher[0]
        if higher[0] <= lower[1]:
            upper_bound = max(lower[1], higher[1])
            merged[-1] = (lower[0], upper_bound)  # replace by merged interval
        else:
            merged.append(higher)    
    
flat_list = [item for sublist in merged for item in sublist]
AlarmTimeSec = pd.DataFrame()
AlarmTimeSec['TimeSec'] = flat_list
AlarmTimeSec['Decision'] = (AlarmTimeSec.index + 1) % 2
A1 = interp1d(AlarmTimeSec['TimeSec'],AlarmTimeSec['Decision'],kind = 'previous',fill_value = 0)


AprilDataRounded['Alarm'] = AprilDataRounded.Inserted_Date.apply(subtime)
AprilDataRounded['Alarm'] = AprilDataRounded.Alarm.apply(A1)

AprilDataAll['Alarm'] = AprilDataAll.Inserted_Date.apply(subtime)
AprilDataAll['Alarm'] = AprilDataAll.Alarm.apply(A1)
#%%

AprilDataRounded['Failure'] = 0
AprilDataAll['Failure'] = 0
for index,value in AlarmDataTrueAlarms.iterrows():
    mask = AprilDataRounded.Inserted_Date.between(AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time'] - datetime.timedelta(minutes = 10),\
                                         AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time'],inclusive = True)
    AprilDataRounded.loc[mask,'Failure'] = 1
    
for index,value in AlarmDataTrueAlarms.iterrows():
    mask = AprilDataAll.Inserted_Date.between(AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time'] - datetime.timedelta(minutes = 10),\
                                         AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time'],inclusive = True)
    AprilDataAll.loc[mask,'Failure'] = 1

Dataset1 = AprilDataRounded[AprilDataRounded.Alarm == 0] #off hours excluded 
Dataset2 = AprilDataAll[AprilDataAll.Alarm == 0]  #all timestamps recorded by the sensors
features = list(set(AprilDataRounded.columns) - set(['Inserted_Date','Alarm','Failure']))
#%%
# Separating out th fe features
x = Dataset1.loc[:,features].values
y = Dataset1.loc[:,'Failure'].values
train_x, test_x, train_y, test_y = train_test_split( x, y, test_size=0.3, random_state=1)

##################base line model####################################
############################################################

scaler = StandardScaler()# Fit on training set only.
scaler.fit(train_x)

# Apply transform to both the training set and the test set.
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)
train_x[np.isnan(train_x)]=10000
test_x[np.isnan(test_x)]=10000

###########################Random forest#####################################

# Building a Random forest model

rf = RandomForestClassifier(n_estimators = 100,random_state=1,oob_score=True,n_jobs=-1)

rffitted = rf.fit(train_x, train_y.ravel())
print(rffitted)

# Use the forest's predict method on the test data
predictions_rf = rf.predict(test_x)
print(predictions_rf)
print(f1_score(test_y,predictions_rf))
print(precision_score(test_y,predictions_rf))
print(recall_score(test_y,predictions_rf))
#%%
# Calculate the absolute errors

#rf_root_mean_square_error=(mean_squared_error(test_y, predictions_rf))**0.5
#print(rf_root_mean_square_error)
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
#rf_most_important = RandomForestRegressor(n_estimators=100, random_state=1)
##
#
#important_features_positions=[]
#
###  Select top 5 important features
#for i in range(5):
#    important_features=feature_importances[i][0]
#    index=features.index(important_features)
#    important_features_positions.append(index)
###selecting important features from train data    
#train_x_important=train_x[:,important_features_positions] 
###selecting important features from train data
#test_x_important=test_x[:,important_features_positions] 
#rf_most_important.fit(train_x_important, train_y.ravel());
#predictions_rf_most_important = rf_most_important.predict(test_x_important)

##
##

## Calculate the absolute errors
#rf_root_mean_square_error_most_imp=(mean_squared_error(test_y, 
#                                predictions_rf_most_important))**0.5
#print(rf_root_mean_square_error_most_imp)

#
#%%
#"""Hyper parameter tuning"""
#
### Look at parameters used by our current forest
#print('Parameters currently in use:\n')
#pprint(rf.get_params())
#
### Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 50, stop = 400,num = 10)]
#print(n_estimators)
#
### Number of features to consider at every split
#max_features = ['auto', 'sqrt']
#
### Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
#
### Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10]
#
### Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
#
### Method of selecting samples for training each tree
#bootstrap = [True, False]
#
### Create the random grid
#random_grid = {'n_estimators': n_estimators,
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}
#
#pprint(random_grid)
#
### Use the random grid to search for best hyperparameters
### First create the base model to tune
#rf_for_tuning = RandomForestRegressor()
#
### Random search of parameters, using 3 fold cross validation, 
### search across 100 different combinations, and use all available cores
#rf_random = RandomizedSearchCV(estimator = rf_for_tuning, 
#                            param_distributions = random_grid, 
#                            n_iter = 100, cv = 3, verbose=2, random_state=1)
#
#
### Fit the random search model
#rf_random.fit(train_x, train_y.ravel())
#print(rf_random.best_params_)
#
### finding the best model
#rf_model_best = rf_random.best_estimator_
#print(rf_model_best)
#
## predicting with the test data on best model
#predictions_best = rf_model_best.predict(test_x)
#
#rf_root_mean_square_error_best=(mean_squared_error(test_y,
#                                        predictions_best))**0.5
#                                                  
#print(rf_root_mean_square_error_best)
