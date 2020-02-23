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
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
ss = StandardScaler().fit(X)
X = ss.transform(X)/2
X_test = ss.transform(X_test)/2

c,r = np.mgrid[[slice(-1,1,0.02)]*2]
p = np.c_[c.flat,r.flat]

def hinge_l(y_true,y_pred):
    return np.maximum(1-y_true*y_pred,0.)

clf = SGDClassifier(loss='hinge',max_iter=1,verbose=1,shuffle=True,random_state=1012,
                   penalty='none')
clf.fit(X,y)
hinge_loss_log = []
for i in range(1000):
    hinge_loss_log.append(hinge_l(y,(clf.coef_@X.T+clf.intercept_).squeeze()))
    clf.partial_fit(X,y)
vmin=np.min(hinge_loss_log[0])
vmax = np.max(hinge_loss_log[0])
fig,ax = plt.subplots(figsize=[8,5])
loss_map = ax.scatter(*X.T,c=hinge_loss_log[0],vmin=vmin,vmax=vmax,cmap='coolwarm',s=20)
cbar = fig.colorbar(loss_map)
time_text = fig.text(0.05, 0.9, '', fontsize = 12,transform=ax.transAxes)
def update(ite):
    ax.clear()
    if np.max(hinge_loss_log[ite+1])<1:
        vmax=1
        vmin=0
    else:
        vmin = np.min(hinge_loss_log[ite+1])
        vmax = np.max(hinge_loss_log[ite+1])
    loss_map=ax.scatter(*X.T,c=hinge_loss_log[ite+1],vmin=vmin,vmax=vmax,cmap='coolwarm',s=20)
    cbar.on_mappable_changed(loss_map)
    time_text.set_text('%d epoch'%(ite+1))
ani = FuncAnimation(fig,update,frames=500,interval=1)
plt.show()