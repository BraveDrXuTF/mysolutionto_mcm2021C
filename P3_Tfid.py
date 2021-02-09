from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# import eli5
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import focalloss
df_id_status = pd.read_excel('/home/xutengfei/meisai2021/2021_MCM_Problem_C_Data/2021MCMProblemC_DataSet.xlsx')
df_filename_id = pd.read_excel('/home/xutengfei/meisai2021/2021_MCM_Problem_C_Data/2021MCM_ProblemC_ Images_by_GlobalID.xlsx')

df_id_status.loc[df_id_status['Lab Status'] == 'Positive ID','Lab Status']=1
df_id_status.loc[df_id_status['Lab Status'] == 'Negative ID','Lab Status']=0
df_word=df_id_status.dropna()
df_word=df_word.drop(index=df_word[df_word['Notes']==' '].index)
df_word=df_id_status.loc[(df_id_status['Lab Status'] == 1)|(df_id_status['Lab Status'] == 0),['Notes','Lab Status']]

vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'[a-zA-Z]{5,}',
            ngram_range=(1,1),
            min_df=1,
            max_df=0.7
            ) # min_df can hurt training
df_word['Notes']=df_word['Notes'].astype(str) # datetime was mixed with notes 
overview_text = vectorizer.fit_transform(df_word['Notes'])
vectorizer.get_feature_names()
x_train=overview_text.toarray()
y_train=df_word['Lab Status'].values
y_train=y_train.astype('int')



 
# linreg = LinearRegression()
# linreg.fit(overview_text, train['log_revenue'])
# eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')
# print('Target value:', train['log_revenue'][1000])
# eli5.show_prediction(linreg, doc=train['overview'].values[1000], vec=vectorizer)


net = nn.Sequential(
    nn.Linear(2243, 7),
    nn.ReLU(),
    nn.Linear(7, 1),
    nn.Sigmoid()
)
cost = focalloss.FocalLoss(alpha=0.95)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
max_epoch = 300
iter_loss = []
batch_loss = []

# del all 0 rows
mask = np.all(np.isnan(x_train), axis=1) | np.all(x_train == 0, axis=1)
x_train = x_train[~mask]
y_train = y_train[~mask]


# for i in range(max_epoch):
#     for n in range(x_train.shape[0]):
#         input = Variable(torch.FloatTensor(x_train[n, :]))
#         output = Variable(torch.FloatTensor(y_train[n]))

#         predict = net(input)
#         loss = cost(predict, output)
#         batch_loss.append(loss.data.numpy())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(loss.data.numpy())
#     iter_loss.append(np.average(np.array(batch_loss)))
# torch.save({
#             'model_state_dict': net.state_dict(),
#         }, '/home/xutengfei/meisai2021/P3_net.pth')

# import matplotlib.pyplot as plt

# x = np.arange(max_epoch)
# y = np.array(iter_loss)
# plt.plot(x, y)
# plt.title('loss curve')
# plt.xlabel('num of iteration')
# plt.ylabel('ave loss')
# plt.show()
