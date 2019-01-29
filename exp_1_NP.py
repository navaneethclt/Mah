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

xls = pd.ExcelFile('May.xlsx',index = False);
#%%

AprilData = xls.parse('Sheet1', index =False, na_values=['NA']);
#%%
AprilData = AprilData.reindex(index = AprilData.index[::-1])
AprilData.reset_index(drop=True, inplace=True)
AprilData = AprilData.dropna(axis = 1);
AprilData = AprilData[AprilData['Parameter Value']< 30000]
AprilData['Machine Parameter'] = AprilData['Machine Parameter'].str.replace(" ","_")
AprilData = AprilData[AprilData['Machine Parameter']!= 'SPRAY_PUMP_MOTOR__A_TEMPERATURE']
SensorNames = AprilData.drop_duplicates('Machine Parameter');

#sdf = AprilData.groupby(['Machine Parameter'],group_keys = True)
#params = SensorsNames.groupby(['Machine Parameter']).groups.keys();
#AprilData[AprilData['Machine Parameter'] == 'TROLLY MOTOR CURRENT']
params = SensorNames.groupby(['Machine Parameter']).groups.keys();
AprilData['Inserted Date'] = AprilData['Inserted Date'].dt.round('1s')
AprilDataM = AprilData.copy()
AprilDataM['Inserted Date'] = AprilDataM['Inserted Date'].dt.round('1min')
#AprilData[AprilData['Machine Parameter'] == 'TROLLY MOTOR CURRENT']

companies = list(params)
i = 1
for c in companies:
         a = 'df' + str(i)
         b = 'temp'+str(i)
    
         exec(a + "= AprilData[AprilData['Machine Parameter'] == c]")
         exec(a + " = "+a+".groupby('Inserted Date',as_index =True)['Parameter Value'].mean()")
         i = i + 1
i = 1
for c in companies:
         a = 'dfm' + str(i)
         b = 'tempm'+str(i)
    
         exec(a + "= AprilDataM[AprilDataM['Machine Parameter'] == c]")
         exec(a + " = "+a+".groupby('Inserted Date',as_index =True)['Parameter Value'].mean()")
         i = i + 1
         #exec(b+"="+a+".resample('7s',on='Inserted Date').last()")
        
#forward filling is questionable since we dont want data preceding a failure to be propagated throughout the dataset
         #exec(b+".fillna(method='ffill',inplace=True)") 
        
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
    
timeindexm = AprilDataM['Inserted Date'].drop_duplicates()
timeindexm.sort_values(ascending=True, kind='quicksort', na_position='last', inplace=True)
working_timem = []

for date in timeindexm:
    if date.hour>=8 and (date.hour<=16 and date.minute<=30):
        if not(date.hour==11 and date.minute>=30):
            if not(date.hour==15 and date.minute<=7):
               working_timem.append(date)
AprilDataRoundedM = pd.DataFrame(index = working_timem , columns = companies)
AprilDataRoundedM['Inserted_Date'] = working_timem
AprilDataAllM = pd.DataFrame(index = timeindexm , columns = companies)
AprilDataAllM['Inserted_Date'] = list(timeindexm)
for i in range(len(companies)):
    b = 'dfm' + str(i+1)
    sensor = companies[i]
    exec("AprilDataRoundedM['"+sensor+"']="+b+"")
    exec("AprilDataAllM['"+sensor+"']="+b+"")
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
                                   
for index,value in AlarmData.iterrows():
    if AlarmData.loc[index,'Alarm_Name'] in set(AlarmClassList):
        AlarmData.loc[index,'Priority'] =  AlarmClass.loc[AlarmClass.Alarm_Name==AlarmData.loc[index,'Alarm_Name'],'Priority'].iloc[0]
for index,value in AlarmData.iterrows():
    if AlarmData.loc[index,'Alarm_Name'] in set(AlarmClassList):
        AlarmData.loc[index,'Alarm Source'] =  AlarmClass.loc[AlarmClass.Alarm_Name==AlarmData.loc[index,'Alarm_Name'],'Alarm Source'].iloc[0]

AlarmData['T/F'] = 'F'
AlarmPriorityGroup= AlarmData.groupby('Priority',group_keys = True)   
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
fig,ax = plt.pyplot.subplots()
for index,value in AlarmDataTrueAlarms.iterrows():
        ax.axvspan(AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time'],AlarmDataTrueAlarms.loc[index,'Alarm_Stop_Time'], alpha=0.2, color='yellow')
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

AprilDataRoundedM['Alarm'] = AprilDataRoundedM.Inserted_Date.apply(subtime)
AprilDataRoundedM['Alarm'] = AprilDataRoundedM.Alarm.apply(A1)

AprilDataAllM['Alarm'] = AprilDataAllM.Inserted_Date.apply(subtime)
AprilDataAllM['Alarm'] = AprilDataAllM.Alarm.apply(A1)
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
    
AprilDataRoundedM['Failure'] = 0
AprilDataAllM['Failure'] = 0
for index,value in AlarmDataTrueAlarms.iterrows():
    mask = AprilDataRoundedM.Inserted_Date.between(AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time'] - datetime.timedelta(minutes = 10),\
                                         AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time'],inclusive = True)
    AprilDataRoundedM.loc[mask,'Failure'] = 1
    
for index,value in AlarmDataTrueAlarms.iterrows():
    mask = AprilDataAllM.Inserted_Date.between(AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time'] - datetime.timedelta(minutes = 10),\
                                         AlarmDataTrueAlarms.loc[index,'Alarm_Start_Time'],inclusive = True)
    AprilDataAllM.loc[mask,'Failure'] = 1

Dataset1 = AprilDataRounded[AprilDataRounded.Alarm == 0]
Dataset2 = AprilDataAll[AprilDataAll.Alarm == 0]

Dataset1m = AprilDataRoundedM[AprilDataRoundedM.Alarm == 0]
Dataset2m = AprilDataAllM[AprilDataAllM.Alarm == 0]

features = list(set(AprilDataRounded.columns) - set(['Inserted_Date','Alarm','Failure']))
#%%
# Separating out the features
x = Dataset1m.loc[:, features].values
y = Dataset.loc[:,'Failure'].values
train_x, test_x, train_y, test_y = train_test_split( x, y, test_size=0.3, random_state=1)
