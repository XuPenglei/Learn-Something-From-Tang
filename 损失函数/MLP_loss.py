from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hinge_loss,log_loss
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def softmax(X):
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]
    return X
def log_loss(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-10, 1 - 1e-10)

    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.sum(y_true * np.log(y_prob),axis=-1)

np.random.seed(1024)
iris = datasets.load_iris()
X=iris.data
X=StandardScaler().fit_transform(X)
y=iris.target
X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
MLP = MLPClassifier(activation='relu',hidden_layer_sizes=(2,2),verbose=1,max_iter=8000,
                    solver='sgd',random_state=10,
                    tol=1e-8,early_stopping=True)
MLP.fit(X,y)
plt.plot(MLP.loss_curve_)
plt.plot(MLP.validation_scores_)
plt.show()
y_01 = to_categorical(y,3)
y_pred = MLP.predict_proba(X)
loss = log_loss(y_01,y_pred)
fig,ax=plt.subplots(2,1)
ax=ax.flatten()
scat1 = ax[0].scatter(*X[:,:2].T,c=loss,cmap='coolwarm')
scat2 = ax[1].scatter(*X[:,:2].T,c=y,cmap='jet')
c1=fig.colorbar(scat1,ax=ax[0],label='损失')
c2=fig.colorbar(scat2,ax=ax[1],ticks=[0,1,2],label='类别')
plt.show()