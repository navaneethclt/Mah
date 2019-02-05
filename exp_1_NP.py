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
         d = 'dfs'+ str(i)
         b = 'temp'+str(i)
    
         exec(a + "= AprilData[AprilData['Machine Parameter'] == c]")
         exec(d + "= AprilData[AprilData['Machine Parameter'] == c]")
         exec(a + " = "+a+".groupby('Inserted Date',as_index =True)['Parameter Value'].mean()")
         i = i + 1
i = 1
for c in companies:
         a = 'dfm' + str(i)
         b = 'tempm'+str(i)
         d = 'dfsm'+ str(i)
    
         exec(a + "= AprilDataM[AprilDataM['Machine Parameter'] == c]")
         exec(d + "= AprilData[AprilData['Machine Parameter'] == c]")
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
xlscat = pd.ExcelFile('Crankcase CM Data-30.01.2019.xlsx',index = False);
#%%
AlarmData = xlsal.parse('Alarm data', index =False, na_values=['NA']);
AlarmClass = xlsalcls.parse('Fault Categories-CCCM (2)', index =False, na_values=['NA']);
AlarmClass.Priority = AlarmClass.Priority.str.capitalize()
AlarmNames = AlarmData.drop_duplicates('Alarm_Name')
AlarmGroup = AlarmData.groupby('Alarm_Name',group_keys = True)
AlarmClassList = AlarmClass.Alarm_Name

AlarmCount = AlarmData.groupby(['Alarm_Name']).size().reset_index(name='count')
AlarmCat = xlscat.parse('Fault Categories-CCCM', skiprows = 2,index =False, na_values=['NA']);
AlarmCat.dropna(inplace = True,axis = 1)
AlarmCatList = AlarmCat .Alarm_Name
for index,value in AlarmData.iterrows():
    if AlarmData.loc[index,'Alarm_Name'] in set(AlarmCatList):
        AlarmData.loc[index,'Chamber'] =  AlarmCat.loc[AlarmCat.Alarm_Name==AlarmData.loc[index,'Alarm_Name'],'Chamber'].iloc[0]                                   
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
        mask = (group['Down_Time'] <= datetime.time(minute = 30)) & (group['Down_Time'] >= datetime.time(minute =1))
        mask = mask[mask == True]
        AlarmData.loc[mask.index,'T/F'] = 'T'
    if(key == 'Breakdown'):
        mask = (group['Down_Time'] <= datetime.time(minute = 30)) & (group['Down_Time'] >= datetime.time(minute = 1))
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
cxls = pd.ExcelFile('MayCycle.xlsx',index = False);
cstartch1 = cxls.parse('Sheet1',index =False, na_values=['NA']);
cstopch1 = cxls.parse('Sheet2',index =False, na_values=['NA']);
cstartch2 = cxls.parse('Sheet3',index =False, na_values=['NA']);
cstopch2 = cxls.parse('Sheet4',index =False, na_values=['NA']);
#%%
cstartch1['stat'] = 1
cstopch1['stat'] = 0
cstartch2['stat'] = 1
cstopch2['stat'] = 0
cstopch1 = cstopch1[cstopch1['Inserted_Date'] <= '2018-05-30 18:13:19.903000']
cdatach1 = pd.DataFrame()
cdatach2 = pd.DataFrame()
cdatach1 = cstartch1.append(cstopch1)
cdatach2 = cstartch2.append(cstopch2)
cdatach1.sort_values('Inserted_Date',inplace = True)
cdatach1.reset_index(drop = True,inplace = True)
cdatach2.sort_values('Inserted_Date',inplace = True)
cdatach2.reset_index(drop = True,inplace = True)
#%%
#cdata = cdata[cdata['Inserted_Date'] <= '2018-05-05 18:13:19.903000']
xmin = cdatach1.loc[0,'Inserted_Date']
ymin = cdatach1.loc[0,'stat']
cycletimech1 = []
fig,ax = plt.pyplot.subplots()


#minutes = plt.dates.MinuteLocator()      
#ax.xaxis.set_minor_locator(minutes)

for index,value in cdatach1.iterrows():
    xmax = cdatach1.loc[index,'Inserted_Date']
    ymax = cdatach1.loc[index,'stat']
    cycle = xmax - xmin
    if(ymin != ymax):
      if (((ymin == 1)&(ymax == 0)) & (cycle < datetime.timedelta(minutes = 120))):
        cycletimech1.append(cycle)
        ax.axvspan(xmin,xmax, alpha=0.2, color='yellow')
      xmin = cdatach1.loc[index,'Inserted_Date']
      ymin = cdatach1.loc[index,'stat']

xmin = cdatach2.loc[0,'Inserted_Date']
ymin = cdatach2.loc[0,'stat']
cycletimech2 = []

for index,value in cdatach2.iterrows():
    xmax = cdatach2.loc[index,'Inserted_Date']
    ymax = cdatach2.loc[index,'stat']
    cycle = xmax - xmin
    if(ymin != ymax):
      if (((ymin == 1)&(ymax == 0)) & (cycle < datetime.timedelta(minutes = 120))):
        cycletimech2.append(cycle)
        ax.axvspan(xmin,xmax, alpha=0.2, color='green')
      xmin = cdatach2.loc[index,'Inserted_Date']
      ymin = cdatach2.loc[index,'stat']

plt.pyplot.plot(dfs1['Inserted Date'],dfs1['Parameter Value'])    

#for index,value in dfs7.iterrows():
# plt.pyplot.scatter(dfs7.loc[index,'Inserted Date'],dfs7.loc[index,'Parameter Value'],s = 1,color = 'red');
#fig,ax = plt.pyplot.subplots()
texts = []     
for index,value in AlarmDataTrueAlarmsch1ch2.iterrows():
        ax.axvspan(AlarmDataTrueAlarmsch1ch2.loc[index,'Alarm_Start_Time'],AlarmDataTrueAlarmsch1ch2.loc[index,'Alarm_Stop_Time'], alpha=0.2, color='red')
        texts.append(ax.text(AlarmDataTrueAlarmsch1ch2.loc[index,'Alarm_Start_Time'],1,AlarmDataTrueAlarmsch1ch2.loc[index,'Alarm_Name']))
adjust_text(texts, only_move='y', arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
ax.set_xlim(['2018-05-01','2018-05-18'])
#%%
def subtime(x):
    return (x - datetime.datetime(2018,4,1)).total_seconds()
#%%
Timelist = []

AlarmDataTrueAlarms = AlarmData[AlarmData['T/F'] == 'T']
AlarmDataTrueAlarmsch1ch2 = AlarmDataTrueAlarms[AlarmDataTrueAlarms['Chamber']!= 'CH3']

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
