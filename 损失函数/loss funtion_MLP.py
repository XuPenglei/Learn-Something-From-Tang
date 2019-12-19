from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap,Colormap
from matplotlib import cm
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#coding: utf-8
np.random.seed(13)

#可视化训练过程损失，捕获控制台输出
# class TextArea(object):
#
#     def __init__(self):
#         self.buffer = []
#
#     def write(self, *args, **kwargs):
#         self.buffer.append(args)

import sys
#
# stdout = sys.stdout
# sys.stdout = TextArea()
X,y= datasets.make_moons(n_samples = 2000, noise=.1)
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

# for i in range(12,25):
#     mlp = MLPClassifier(hidden_layer_sizes=(i),verbose=False,tol = 0.00000001,n_iter_no_change =10000,max_iter=2000)
#     mlp.fit(X_train, y_train)
#
#     #可视化训练过程损失
#     # text_area, sys.stdout = sys.stdout, stdout
#     #Loss = []
#     # for i in range(0,text_area.buffer.__len__()-2,2):
#     #     epochi =text_area.buffer[i]
#     #     loss = float(epochi[0][-10:])
#     #     Loss.append(loss)
#     plt.plot(mlp.loss_curve_,label = str(i))
#     # plt.title('(i)')
#     # plt.xlabel('interation')
#     # plt.ylabel('loss')
#     # plt.show()
#     # print (mlp.score(X_test,y_test))
#     # print (mlp.n_layers_)
#     # print (mlp.n_iter_)
#     # print (mlp.loss_)
#     # print (mlp.out_activation_)
#     print(i)
# plt.title('不同隐层节点损失曲线差异')
# plt.xlabel('interation')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

mlp = MLPClassifier(hidden_layer_sizes=(2,2,2,2,2,2,2),verbose=True,tol = 0.00000001,n_iter_no_change =10000,max_iter=10000)
mlp.fit(X_train, y_train)
plt.plot(mlp.loss_curve_)
plt.title('(2,2,2,2,2,2,2)')
plt.xlabel('interation')
plt.ylabel('loss')
plt.show()
