import pandas as pd
df_id_status = pd.read_excel('2021MCMProblemC_DataSet.xlsx',index_col=0)
df_filename_id = pd.read_excel('2021MCM_ProblemC_ Images_by_GlobalID.xlsx')
df_filename_status = df_filename_id
df_filename_status['Lab Status']=''
for index,row in df_filename_id.iterrows():
	df_filename_status.loc[index,'Lab Status']=df_id_status[df_id_status.index.str.contains(row['GlobalID'])]['Lab Status'].values[0]
df_filename_status.to_csv('info.csv')
    

