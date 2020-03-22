from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hinge_loss,log_loss
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from sklearn import datasets
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

X,y = datasets.make_moons(n_samples=2000, noise=.1)
y = np.where(y==0,-1,1)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
ss = StandardScaler().fit(X)
X = ss.transform(X)/2
X_test = ss.transform(X_test)/2

c,r = np.mgrid[[slice(-1,1,0.02)]*2]
p = np.c_[c.flat,r.flat]

def hinge_l(y_true,y_pred):
    return np.maximum(1-y_true*y_pred,0.)
def squared_hinge_l(y_true,y_pred):
    return np.square(np.maximum(1-y_true*y_pred,0.))
def modified_huber_l(y_true,y_pred):
    return np.where(y_pred*y_true>=-1,np.square(np.max(0,1-y_true*y_pred)),-4*y_pred*y_true)
def log_l(y_true,y_pred):
    return np.log(1+np.exp(-y_true*y_pred))
def perceptron_l(y_true,y_pred,t=0):
    """ 损失函数不对 """
    return np.where(y_pred*y_true<=t,-(y_pred*y_true),0)

loss_name='perceptron'
if loss_name=='hinge':
    loss_f=hinge_l
elif loss_name=='squared_hinge':
    loss_f=squared_hinge_l
elif loss_name=='modified_huber':
    loss_f=modified_huber_l
elif loss_name=='log':
    loss_f=log_l
elif loss_name=='perceptron':
    loss_f = perceptron_l
else:
    raise ValueError
clf = SGDClassifier(loss=loss_name,max_iter=1,verbose=1,shuffle=True,random_state=1012,
                   penalty='none')
clf.fit(X,y)
loss_log = []
p_result_log = []
for i in range(1000):
    loss_t = loss_f(y,(clf.coef_@X.T+clf.intercept_).squeeze())
    loss_log.append(loss_t)
    p_result_log.append(clf.decision_function(p))
    print(np.mean(loss_t,axis=-1))
    clf.partial_fit(X,y)
s = plt.scatter(*X.T,c=(clf.coef_@X.T+clf.intercept_).squeeze(),cmap='coolwarm')
plt.colorbar(s)
plt.show()

vmin=np.min(loss_log[-1])
vmax = np.max(loss_log[-1])
fig,ax = plt.subplots(figsize=[8,5])
m1 = ax.contourf(p[:,0].reshape(100,100), p[:,1].reshape(100,100), p_result_log[0].reshape(100,100),cmap = 'jet')
levels = [-1, 0.0, 1]  # 将输出值分为-1,0,1几个区间
linestyles = ['dashed', 'solid', 'dashed']
m3 = ax.contour(p[:,0].reshape(100,100), p[:,1].reshape(100,100), p_result_log[0].reshape(100,100),
                levels,colors='black',linestyles= linestyles)
clabel = plt.clabel(m3, inline=True, fontsize=10)  # 添加文字标签 inlins表示等高线是穿过数字还是不穿过
loss_map = ax.scatter(*X.T,c=loss_log[0],vmin=vmin,vmax=vmax,cmap='coolwarm',s=20)
cbar = fig.colorbar(loss_map)
time_text = fig.text(0.05, 0.9, '', fontsize = 12,transform=ax.transAxes)
def update(ite):
    ax.clear()
    m1 = ax.contourf(p[:,0].reshape(100,100), p[:,1].reshape(100,100),
                     p_result_log[ite+1].reshape(100,100), cmap='jet')
    m3 = ax.contour(p[:,0].reshape(100,100), p[:,1].reshape(100,100), p_result_log[ite+1].reshape(100,100),
                    levels, colors='black', linestyles=linestyles)
    print(np.mean(loss_log[ite+1]))
    loss_map = ax.scatter(*X.T, c=loss_log[ite + 1], cmap='coolwarm', s=20)
    clabel = plt.clabel(m3, inline=True, fontsize=10)
    cbar.on_mappable_changed(loss_map)
    time_text.set_text('%s %d epoch'%(loss_name,ite+1))
ani = FuncAnimation(fig,update,frames=500,interval=1)
plt.show()