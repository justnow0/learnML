import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat 
from scipy.optimize import minimize

#加载数据集

def load_data(path):
    data = loadmat(path)
    x = data['X']
    y = data['y']
    return x, y

x,y = load_data('ex3data1.mat')
'''
print(np.unique(y))
print(x.shape, y.shape)
print(y)
'''
# Visualizing the data 随机打印一个数字
def plot_an_image(x):
    pick_one = np.random.randint(0,5000)
    image = x[pick_one,:] # 选择该行的所有内容
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap='gray_r')
    plt.xticks([])  # 去除刻度，美观
    plt.yticks([])
    plt.show()
    print('this should be {}'.format(y[pick_one]))


def plot_100_image(X):
    """
    随机画100个数字
    """
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 随机选100个样本
    sample_images = X[sample_idx, :]  # (100,400)
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10 * row + column].reshape((20, 20)),cmap='gray_r')
    plt.xticks([])
    plt.yticks([])        
    plt.show()

#plot_100_image(x)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def regularized_cost(theta, x, y , l):
    """
    不惩罚theta_0
    args :
        x: feature matrix, (m, n+1) 插入了x0=1
        y: target vector, (m,)
        l: lambda constant for regularization
    """
    thetaReg = theta[1:]
    first = (-y * np.log(sigmoid(x@theta))) + (y-1)\
        *np.log(1-sigmoid(x@theta))
    reg = (thetaReg@thetaReg)*l / (2*len(x))

    return np.mean(first) + reg 

def regularized_gradient(theta, x, y ,l):
   
    """
    don't penalize theta_0
    args:
        l: lambda constant
    return:
        a vector of gradient 
    """
    thetaReg = theta[1:]
    first = (1 / len(x)) * x.T@ (sigmoid(x@theta) - y)
    # 这里人为插入一维0, 使得对theta_0不惩罚,方便计算
    reg = np.concatenate([np.array([0]), (l/len(x))*thetaReg])
    return (first + reg)


def one_vs_all(x, y, l, k):
    """
    generalized logistic regression
    args:
        x: feature matrix, (m, n+1) #with incercept x0=1
        y: target vector,(m,)
        l: lambda constant for regularization
        k: number of labels
    return trained parameters 
    """
    all_theta = np.zeros((k, x.shape[1])) #(10, 401)
    for i in range(1, k+1):
        """
        当i为1时，此时y_i表示的是所有5000个图片的类别是否为1，若是1，则返回1，否则返回0。
        
        """
        theta = np.zeros(x.shape[1])
        #y是5000个图像的真实标签，即类别
        print("y的内容是",y,"\n","y的shape是:",y.shape)
        y_i = np.array([1 if label == i else 0 for label in y])
        print("y_i的内容是",y_i)
        ret = minimize(fun=regularized_cost, x0=theta, args=(x,y_i,l),method = 'TNC',\
            jac = regularized_gradient,options={'disp':True})
        print(ret.x)
        all_theta[i-1,:] = ret.x
        """ 
        当i=1是，此时ret.x表示的是通过正则化梯度下降函数计算出的最小损失函数对应的theta;!!
        注意，此时all_theta[0,:],即第一行数据为401个theta值(theta_1~the_401)；
        最终，all_theta表示的是10行不同的theta值
        """
    return all_theta

def predict_all(x, all_theta):
    """
    通过计算假设函数，来判断最大的可能性是什么数据
    x的维度是(5000,401)，all_theta的维度是(10,401)
    所以x@all_theta.T是(5000,10)
    可以这样理解
    我们先用x中的第一行与all_theta.T的第一列相乘即:
    x_0*theta_0+x_1*theta_1........
    h
    也就是说:第一个图片样本的特征值与第一类图片通过求最小的损失函数的401个theta值相乘。
    将其带入到sigmoid函数中得到，得到假设函数
    然后再用x中的第一行与all_theta.T的第二列相乘即：
    第一个图片样本的特征值与第二类的图片通过计算的theta相乘，再次得到一个假设函数。
    这样我们，依次将第一个图片的特征值与10个类型的theta值相乘。
    最后再在其中找到最大的值，即可将该图片判断为此类。

    接着，其他的4999个图片特征值，每个都有10个计算的假设函数值。
    最终得到的是一个(5000x10)的二维数组，其中


    
    """
    h = sigmoid(x @ all_theta.T)
    print("h的值是",h)
    print("h的维度是",h.shape)
    h_argmax = np.argmax(h, axis=1)

    print("test",h_argmax,"test")
    #这里面要注意的是从1开始的，所以要将索引值加1
    h_argmax = h_argmax + 1
    print("a",h_argmax,"a")
    return h_argmax

#raw_y中的内容是每个图像的标签,即类别
raw_x, raw_y = load_data('ex3data1.mat')
print("哈",raw_y,"哈")
x = np.insert(raw_x, 0, 1, axis=1)
y = raw_y.flatten()

all_theta = one_vs_all(x,y,1,10)
y_pred = predict_all(x, all_theta)

#与原有标签进行对比，才能知道是否准确
accuracy = np.mean(y_pred == y)
"""
y_pred
为我们预测的类型5000个值
y是真实的标签值
"""
print('accuracy = {0}%'.format(accuracy * 100))

