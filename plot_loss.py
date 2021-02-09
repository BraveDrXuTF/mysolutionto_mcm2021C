import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('/home/xutengfei/meisai2021/results.csv')
plt.figure()
plt.plot(df.iloc[:,0])
plt.show()
plt.figure()
plt.plot(df.iloc[:,2])
plt.show()