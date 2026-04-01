import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.utils.helpers import mode

def reshape_samples(samples, length=50):
    """重构样本数据"""
    EMGs,labels=samples
    reshape_EMGs=np.array([EMGs[:,0:length]])
    reshape_labels=np.array([mode(labels[0:length])])
    for i in range(labels.shape[0]-int(length/10)):
        label=np.array([mode(labels[i+1:i+1+int(length/10)])])
        EMG=np.array([EMGs[:,i*10+10:i*10+10+length]])
        reshape_labels=np.concatenate((reshape_labels,label),axis=0)
        reshape_EMGs=np.concatenate((reshape_EMGs,EMG),axis=0)
    reshape_samples=(reshape_EMGs,reshape_labels)
    return reshape_samples

def preparing(samples, test_size=0.2):
    """数据预处理"""
    X,y=samples
    label_set=set(y)
    label_dic={}
    for i in range(len(label_set)):
        label_dic[list(label_set)[i]]=i
    for j in range(len(y)):
        y[j]=label_dic[y[j]]
    y_categorical = to_categorical(y)  # one-hot 编码标签
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=test_size, random_state=28)
    return X_train, X_test, y_train, y_test, label_dic