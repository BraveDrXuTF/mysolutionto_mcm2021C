import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
data_id_status = pd.read_excel('2021MCMProblemC_DataSet.xlsx')
data_jingwei=data_id_status.loc[(data_id_status['Lab Status'] == 'Positive ID')|(data_id_status['Lab Status'] == 'Negative ID'),['Latitude','Longitude','Lab Status']]
data_jingwei.loc[data_jingwei['Lab Status'] == 'Positive ID','Lab Status']=1
data_jingwei.loc[data_jingwei['Lab Status'] == 'Negative ID','Lab Status']=0

data_jingwei_negativesample=data_jingwei[data_jingwei['Lab Status']==0].sample(n=14)
data_jingwei=data_jingwei[data_jingwei['Lab Status']==1].append(data_jingwei_negativesample)
data_jingwei=data_jingwei.sample(frac=1)

fig1=plt.figure()
plt.scatter(data_jingwei[data_jingwei['Lab Status']==0]['Longitude'],data_jingwei[data_jingwei['Lab Status']==0]['Latitude'],s=1,color='b',marker='^')
plt.scatter(data_jingwei[data_jingwei['Lab Status']==1]['Longitude'],data_jingwei[data_jingwei['Lab Status']==1]['Latitude'],s=20,color='r')
plt.show()

x_train=data_jingwei.loc[:,['Latitude','Longitude']].values
y_train=data_jingwei['Lab Status'].values
y_train=y_train.astype('int')
# x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1, train_size=0.6)


def plot_dataset(X,y, axes):
    plt.plot( X[:,1][y==0], X[:,0][y==0], "bs" )
    plt.plot( X[:,1][y==1], X[:,0][y==1], "r^" )
    plt.legend(['Negative','Positive'])
    # plt.axis( axes )
    plt.grid( True, which="both" )
    

# contour函数是画出轮廓，需要给出X和Y的网格，以及对应的Z，它会画出Z的边界（相当于边缘检测及可视化）
def plot_predict(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid( x0s, x1s )
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict( X ).reshape( x0.shape )
    y_decision = clf.decision_function( X ).reshape( x0.shape )
    plt.contour( x1, x0, y_pred, cmap=plt.cm.winter, alpha=0.5 )
    C=plt.contour( x1, x0, y_decision, cmap=plt.cm.spring, alpha=0.5 )
    plt.clabel(C, inline=True, fontsize=10,colors='black')

svm = Pipeline([
                                ("scaler", StandardScaler()), 
                                ("svm_clf", SVC(kernel="rbf", gamma=1, C=1))
                            ])
# svm=SVC(kernel="linear")
fig2=plt.figure(figsize=(6,3))
svm.fit( x_train,y_train )
plot_dataset( x_train,y_train, [45,50,-125,-115] )
plot_predict( svm, [45,50,-125,-115] )
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()







# use knn to train
# 模型训练
k = 5
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(x_train, y_train)

# 进行预测
X_sample = np.array([[48.75,-122.7]])
y_sample = clf.predict(X_sample)

neighbors=clf.kneighbors(X_sample, return_distance=False)
#取出来的点是训练样本X里的索引

# 画出示意图
plt.figure(figsize=(8,5))
# plt.xlim((-116,-125))
# plt.ylim(((45,50)))
plt.scatter(data_jingwei[data_jingwei['Lab Status']==0]['Longitude'],data_jingwei[data_jingwei['Lab Status']==0]['Latitude'],s=20,color='b',marker='^')
plt.scatter(data_jingwei[data_jingwei['Lab Status']==1]['Longitude'],data_jingwei[data_jingwei['Lab Status']==1]['Latitude'],s=20,color='r')
plt.scatter(X_sample[0][1],X_sample[0][0],s=50,color='y',marker='*')
for i in neighbors[0]:
    plt.plot([x_train[i][1], X_sample[0][1]], [x_train[i][0], X_sample[0][0]], 
             '-.', linewidth=0.6);    # 预测点与距离最近的 5 个样本的连线
plt.show()




# stacking&blending

data_id_status = pd.read_excel('2021MCMProblemC_DataSet.xlsx')
data_jingwei=data_id_status.loc[(data_id_status['Lab Status'] == 'Positive ID')|(data_id_status['Lab Status'] == 'Negative ID'),['Latitude','Longitude','Lab Status']]
data_jingwei.loc[data_jingwei['Lab Status'] == 'Positive ID','Lab Status']=1
data_jingwei.loc[data_jingwei['Lab Status'] == 'Negative ID','Lab Status']=0

x_train=data_jingwei.loc[:,['Latitude','Longitude']].values
y_train=data_jingwei['Lab Status'].values
y_train=y_train.astype('int')

y_tp=svm.predict(x_train)
acc_svm=np.sum((y_tp==y_train).astype(float))/x_train.shape[0]
print(acc_svm)
y_tp=clf.predict(x_train)
acc_knn=np.sum((y_tp==y_train).astype(float))/x_train.shape[0]
print(acc_knn)
print(acc_svm/(acc_svm+acc_knn),acc_knn/(acc_knn+acc_svm))
# final goal = 0.5083783783783784 0.49162162162162165