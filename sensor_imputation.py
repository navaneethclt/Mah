# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:25:36 2019

@author: SENAIIITM
"""

import pandas as pd
import itertools
from pandas import ExcelWriter
from pandas import ExcelFile
import xlsxwriter
import matplotlib as plt
import datetime

#creating a datetime64 series of working hours corresponding to November
#accordingly change start and end time for different month data 
working_time = []
dates = pd.date_range('2018/11/01 08:00:05','2018/11/30 16:30:00',freq='7S')
for date in dates:
    if date.hour>=8 and (date.hour<=14 and date.minute<=30):
        if not(date.hour==11 and date.minute>=30):
            if not(date.hour==15 and date.minute<=7):
               working_time.append(date)
working_time = pd.to_datetime(working_time) 

xls = pd.ExcelFile('nov.xlsx',index = False)
#%%

AprilData = xls.parse('Sheet1', skiprows=2, index =False, na_values=['NA'])
#%%

AprilData = AprilData.dropna(axis = 1)
AprilData['Machine Parameter'] = AprilData['Machine Parameter'].str.replace(" ","_")
AprilData = AprilData[['Machine Parameter','Parameter Value','Inserted Date']]
AprilData = AprilData[(AprilData['Parameter Value']<30000)]
SensorNames = AprilData.drop_duplicates('Machine Parameter')


params = SensorNames.groupby(['Machine Parameter']).groups.keys()


companies = list(params)
#%%
i = 1
for c in companies:
         a = 'df' + str(i)
         b = 'temp'+str(i)
    
         exec(a + "= AprilData[AprilData['Machine Parameter'] == c]")
         exec(b+"="+a+".resample('7s',on='Inserted Date').last()")
        
#forward filling is questionable since we dont want data preceding a failure to be propagated throughout the dataset
         #exec(b+".fillna(method='ffill',inplace=True)") 
         i = i + 1
#%%   

new_df = pd.DataFrame(index = working_time , columns = companies)
for i in range(len(companies)):
    b = 'temp' + str(i+1)
    sensor = companies[i]
    exec("new_df['"+sensor+"']="+b+"['Parameter Value']")
   
#%%   
print(new_df)  

