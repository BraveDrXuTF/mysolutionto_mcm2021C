#-*- coding: utf-8 -*-
#arima时序模型

import pandas as pd
import numpy as np
#参数初始化


#读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
data_id_status = pd.read_excel('2021MCMProblemC_DataSet.xlsx')
positive_time=pd.to_datetime(data_id_status[data_id_status['Lab Status'] == 'Positive ID']['Detection Date'])
positive_time=positive_time.sort_values()
positive_dict={"positive_sum":pd.Series(np.arange(1,len(positive_time)+1),index=positive_time.values)}
positive_time=pd.DataFrame(positive_dict)
forecastnum = 5

print(type(positive_time))
print(positive_time)
#时序图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
positive_time.plot()
plt.show()

#自相关图
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(positive_time).show()

#平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
print(u'原始序列的ADF检验结果为：', ADF(positive_time[u'positive_sum']))
#返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

#差分后的结果
D_data = positive_time.diff().dropna()
D_data.columns = ['positive_sum_diff']
D_data.plot() #时序图
plt.show()
plot_acf(D_data).show() #自相关图
from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(D_data).show() #偏自相关图
print(u'差分序列的ADF检验结果为：', ADF(D_data['positive_sum_diff'])) #平稳性检测

#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1)) #返回统计量和p值

from statsmodels.tsa.arima_model import ARIMA

positive_time['positive_sum'] = positive_time['positive_sum'].astype(float)
# #定阶
# pmax = int(len(D_data)/10) #一般阶数不超过length/10
# qmax = int(len(D_data)/10) #一般阶数不超过length/10
# bic_matrix = [] #bic矩阵
# for p in range(pmax+1):
#   tmp = []
#   for q in range(qmax+1):
#     try: #存在部分报错，所以用try来跳过报错。
#       tmp.append(ARIMA(positive_time, (p, 1, q)).fit().bic)
#     except:
#       tmp.append(None)
#   bic_matrix.append(tmp)
#
# bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值

# p,q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
# print(u'BIC最小的p值和q值为：%s、%s' %(p,q))

model = ARIMA(positive_time, (1, 0, 1)).fit() #建立ARIMA(0, 1, 1)模型
model.summary2() #给出一份模型报告
predict_value,wucha,qujian=model.forecast(5) #作为期5天的预测，返回预测结果、标准误差、置信区间。
print(predict_value,wucha,qujian)