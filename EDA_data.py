import pandas as pd
import numpy as np
df_id_status = pd.read_excel('2021MCMProblemC_DataSet.xlsx')
df_filename_id = pd.read_excel('2021MCM_ProblemC_ Images_by_GlobalID.xlsx')
df_filename_status = pd.read_csv('info.csv')

#0 drop nan
df_id_status=df_id_status.dropna(axis=0,how='any') #drop all rows that have any NaN values


#1 find the Positive ID file in 2021MCMProblemC_DataSet.xlsx and info.csv
# print(df_filename_status[df_filename_status['Lab Status']=='Positive ID'])
# print(df_id_status[df_id_status['Lab Status']=='Positive ID'])

#2 plot p,n,u reports scatter
import matplotlib.pyplot as plt
fig1 = plt.figure(1,figsize=(10,8))
plt.scatter(df_id_status[df_id_status['Lab Status']=='Unprocessed']['Longitude'],df_id_status[df_id_status['Lab Status']=='Unprocessed']['Latitude'],s=1,color='k')
plt.scatter(df_id_status[df_id_status['Lab Status']=='Negative ID']['Longitude'],df_id_status[df_id_status['Lab Status']=='Negative ID']['Latitude'],s=1,color='b',marker='^')
plt.scatter(df_id_status[df_id_status['Lab Status']=='Unverified']['Longitude'],df_id_status[df_id_status['Lab Status']=='Unverified']['Latitude'],s=1,color='y',marker='s')
plt.scatter(df_id_status[df_id_status['Lab Status']=='Positive ID']['Longitude'],df_id_status[df_id_status['Lab Status']=='Positive ID']['Latitude'],s=20,color='r')
ax = fig1.gca()
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels = ['Unprocessed','Negative','Unverified','Positive'], loc='upper left')
# plt.scatter([df_id_status['Longitude'],df_id_status[df_id_status['Lab Status']=='Positive ID']['Longitude']],[df_id_status['Latitude'],df_id_status[df_id_status['Lab Status']=='Positive ID']['Latitude']],s=2)
plt.show()
# df_id_status[df_id_status['Lab Status']=='Positive ID']['Latitude']
fig3=plt.figure()
plt.scatter(df_id_status[df_id_status['Lab Status']=='Negative ID']['Longitude'],df_id_status[df_id_status['Lab Status']=='Negative ID']['Latitude'],s=1,color='b',marker='^')
plt.scatter(df_id_status[df_id_status['Lab Status']=='Positive ID']['Longitude'],df_id_status[df_id_status['Lab Status']=='Positive ID']['Latitude'],s=20,color='r')
plt.show()
# # save as csv
# df_filename_id.to_csv('df_filename_id.csv')
# df_id_status.to_csv('df_id_status.csv')


#3 find the Negative ID file in 2021MCMProblemC_DataSet.xlsx and info.csv
# print(df_filename_status.loc[(df_filename_status['Lab Status']=='Negative ID')&(df_filename_status['FileName'].str.endswith('.jpg'))])
# print(df_id_status.loc[(df_id_status['Lab Status']=='Negative ID')&])

#4 plot positive-time
positive_time=pd.to_datetime(df_id_status[df_id_status['Lab Status']=='Positive ID']['Detection Date'])
positive_time=positive_time.sort_values()
# positive_time=positive_time.reset_index(drop=True)
# print(type(positive_time.index))

# df_id_status=df_id_status.set_index('Detection Date')
# negative_time=pd.to_datetime(df_id_status[df_id_status['Lab Status']=='Negative ID'],errors='coerce')
# negative_time=df_id_status[df_id_status['Lab Status']=='Negative ID']['Detection Date']
# negative_time=negative_time.sort_values()
# unverified_time=pd.to_datetime(df_id_status[df_id_status['Lab Status']=='Unverified'][''],errors='coerce')
# unverified_time=df_id_status[df_id_status['Lab Status']=='Unverified']['Detection Date']

unverified_time=df_id_status[df_id_status['Lab Status']=='Unverified']
unverified_time['Detection Date']=pd.to_datetime(unverified_time['Detection Date'],errors='coerce')
unverified_time=unverified_time.sort_values(by=['Detection Date'])
unverified_time=unverified_time[unverified_time['Detection Date'].notnull()] # sort by date and exclude NaT
unverified_time=unverified_time.set_index('Detection Date')
unverified_time=unverified_time['2019':]
print('unverified:')
print(unverified_time)
unverified_time['unverified_sum']=np.ones(unverified_time.shape[0])
unverified_time=unverified_time.resample('M').sum()
print('unverified resampled by month:')
print(unverified_time)
negative_time=df_id_status[df_id_status['Lab Status']=='Negative ID']
negative_time['Detection Date']=pd.to_datetime(negative_time['Detection Date'],errors='coerce')
negative_time=negative_time.sort_values(by=['Detection Date'])
negative_time=negative_time[negative_time['Detection Date'].notnull()] # sort by date and exclude NaT
negative_time=negative_time.set_index('Detection Date')
negative_time=negative_time['2019':] # use data after year 2000
print('negative:')
print(negative_time)
negative_time['negative_sum']=np.ones(negative_time.shape[0])
negative_time=negative_time.resample('M').sum()
fig2=plt.figure(figsize=(8,6))
ax = fig2.gca()
plt.plot(positive_time,np.arange(1,len(positive_time)+1))
plt.plot(negative_time['negative_sum'])
plt.plot(unverified_time['unverified_sum'])
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels = ['Cumulative number of positive reports','Number of new negative reports in single month','Number of new unverified reports in single month'], loc='upper left')
plt.show()
fig3=plt.figure()
ax = fig2.gca()
# positive single month
positive_time=df_id_status[df_id_status['Lab Status']=='Positive ID']
positive_time['Detection Date']=pd.to_datetime(positive_time['Detection Date'],errors='coerce')
positive_time=positive_time.sort_values(by=['Detection Date'])
positive_time=positive_time[positive_time['Detection Date'].notnull()] # sort by date and exclude NaT
positive_time=positive_time.set_index('Detection Date')
positive_time=positive_time['2019':]
positive_time['positive_sum']=np.ones(positive_time.shape[0])
positive_time=positive_time.resample('M').sum()
plt.bar(positive_time.index,positive_time['positive_sum'])
plt.show()