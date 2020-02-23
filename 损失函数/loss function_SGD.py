#===============随机梯度下降法分类===============

from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap,Colormap
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import hinge_loss,log_loss
import math
import time
import scipy.io as sio
import sklearn
import os
#coding: utf-8


#可视化训练过程损失，捕获控制台输出
# class TextArea(object):
#
#     def __init__(self):
#         self.buffer = []
#
#     def write(self, *args, **kwargs):
#         self.buffer.append(args)
#
# import sys
#
# stdout = sys.stdout
# sys.stdout = TextArea()
np.random.seed(13)


#X,y= datasets.make_moons(n_samples = 2000, noise=.1)
X, y = datasets.make_blobs(n_samples=2000, centers=2, random_state=0, cluster_std=0.60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#构建特征空间
c,r = np.mgrid[[slice(X.min()- .2,X.max() + .2,50j)]*2]
p = np.c_[c.flat,r.flat]     #p为二维空间中所有点的坐标（2500，2）

#自定义cmap
bottom = cm.get_cmap('Blues_r', 512)
top = cm.get_cmap('Oranges', 512)
newcolors = np.vstack((bottom(np.linspace(0.5, 1, 512)),
                       top(np.linspace(0, 0.5, 512))))
cm_bright = ListedColormap(newcolors, name='OrangeBlue')
#自定义cmap
bottom = cm.get_cmap('Greens_r', 512)
top = cm.get_cmap('Reds', 512)
newcolors = np.vstack((bottom(np.linspace(0.5, 1, 512)),
                       top(np.linspace(0, 0.75, 512))))
cm_bright1 = ListedColormap(newcolors, name='greenred')
#归一化
ss = StandardScaler().fit(X_train)     #计算均值标准差
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)
p = ss.transform(p)


clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=2000, fit_intercept=True,verbose=True)  #
clf.fit(X_train, y_train)
#可视化训练过程损失
# text_area, sys.stdout = sys.stdout, stdout
# Loss = []
# for i in range(2,text_area.buffer.__len__(),6):
#     epochi =text_area.buffer[i]
#     loss = float(epochi[0][-8:])
#     Loss.append(loss)
# plt.plot(Loss)
# plt.title('perceptron')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
clf.loss_function
X_test_result = clf.predict(X_test)
p_result = clf.decision_function(p)
plt.figure()
m1 = plt.contourf(p[:,0].reshape(50,50), p[:,1].reshape(50,50), p_result.reshape(50,50),cmap = cm_bright1)
m2 = plt.scatter(*X_test.T,c = X_test_result,cmap = cm_bright,edgecolors='white',s = 20,linewidths = 0.5)
levels = [-1, 0.0, 1]  # 将输出值分为-1,0,1几个区间
linestyles = ['dashed', 'solid', 'dashed']
m3 = plt.contour(p[:,0].reshape(50,50), p[:,1].reshape(50,50), p_result.reshape(50,50),levels,colors='black',linestyles= linestyles)
plt.clabel(m3, inline=True, fontsize=10)  # 添加文字标签 inlins表示等高线是穿过数字还是不穿过
plt.colorbar(m1)
cbar = plt.colorbar(m2)
cbar.set_ticks(np.linspace(0, 1, 2))
cbar.set_ticklabels(('0','1'))
plt.title('squared_hinge')
plt.show()





