import numpy as np
from math import sqrt
pi=3.1415927
from scipy.fftpack import fft,ifft
from scipy.signal import welch,stft
import pywt
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import serial
import serial.tools.list_ports
import time
import os
import keyboard
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential # type: ignore
from tensorflow.keras.layers import Input, SimpleRNN, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout, SpatialDropout2D, SpatialDropout1D  # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.constraints import max_norm # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False
#获取标签
def get_lable():
    label=''
    KEY_LIST=['q','w','e','r','t','y','u','i','o','p',
               'a','s','d','f','g','h','j','k','l',
                'z','x','c','v','b','n','m']
    for key in KEY_LIST:
        if keyboard.is_pressed(key):
            label+=key
    if label:
        return label
    else:
        return 'IDLE'
#串口读取(8+6)
def serial_reader_full(ser):
    ser.read_all()
    EMG_samples=[[] for i in range(8)]
    read_flag = 0
    while not read_flag:
        com_input = ser.read(1).hex()
        if com_input=='aa':
            com_input = ser.read(1).hex()
            if com_input=='aa':
                label=get_lable()
                while ser.inWaiting()<96:
                    pass
                read_flag=1
    com_input = ser.read(ser.inWaiting())
    for package_index in range(10):
        for EMG_index in range(8):
            start_index=14+package_index*8+EMG_index
            EMG_samples[EMG_index].append([int(com_input[start_index:start_index+1].hex(),16)-127])
    return np.array([label]),np.array(EMG_samples)
#串口读取(8)
def serial_reader_pure(ser):
    ser.read_all()
    EMG=[[] for i in range(8)]
    read_flag = 0
    while not read_flag:
        com_input = ser.read(1).hex()
        if com_input=='0a':
            label=get_lable()
            while ser.inWaiting()<400:
                pass
            read_flag=1
    com_input = ser.read(ser.inWaiting())
    for package_index in range(10):
        data=str(com_input,encoding='utf-8').split('\n')
        for EMG_index in range(8):
            EMG[EMG_index].append([int(data[package_index].split(' ')[EMG_index])-2048])
    return np.array([label]),np.array(EMG)
#获得众数
def mode(array):
    vals, counts = np.unique(array, return_counts=True)
    index = np.argmax(counts)
    return vals[index]
#samples_storage格式设置
def shape_samples(EMG_storage,EMG_queue,EMG,label_storage,label_queue,label,update_length=50,sample_length=500):
    if EMG_queue.shape[1] < sample_length:
        EMG_queue=np.concatenate((EMG_queue,EMG),axis=1)
        label_queue=np.concatenate((label_queue,label),axis=0)
        return EMG_storage,EMG_queue,label_storage,label_queue
    else:
        mostlike_label=np.array([mode(label_queue)])
        EMG_storage=np.concatenate((EMG_storage,[EMG_queue]),axis=0)
        label_storage=np.concatenate((label_storage,mostlike_label),axis=0)
        EMG_queue=np.concatenate((EMG_queue,EMG),axis=1)
        EMG_queue=EMG_queue[:,-sample_length+update_length-10 :,:]
        label_queue=np.concatenate((label_queue,label),axis=0)
        label_queue=label_queue[-int((sample_length-update_length+10)/10):]
        return EMG_storage,EMG_queue,label_storage,label_queue
#EMG及labels数据保存
def save_samples(filename,samples):
    np.savez(filename, EMG=samples[0], labels=samples[1])
    print(f'Save datas sucessfully!\nSaving file:{filename}')
    return True
#EMG及labels数据读取
def load_samples(filename):
    EMG=np.load(filename)['EMG']
    labels=np.load(filename)['labels']
    print('Load datas sucessfully!')
    samples=(EMG,labels)
    return samples
#收集样本
def Collect_samples(serial_reader,sample_length=500):
    #plt.ion()
    #plt.figure(8,(12.8,6.4))
    with serial.Serial(port="COM4", baudrate=115200, timeout=0.1) as ser:
        #样本存储初始化
        label_queue,EMG_queue=serial_reader(ser)
        for k in range(int(sample_length/10)-1):
            label,EMG=serial_reader(ser)
            EMG_queue=np.concatenate((EMG_queue,EMG),axis=1)
            label_queue=np.concatenate((label_queue,label),axis=0)
        label_storage=np.array([mode(label_queue)])
        EMG_storage=np.array([EMG_queue])
        #样本队列初始化
        label_queue,EMG_queue=serial_reader(ser)
        plot_queue=EMG_queue.copy()
        print('Preparing finish.\nPress \'left Ctrl\' to start collecting samples.\nPress \'left Shift\' to stop collecting samples.')
        p=0
        while True:
            if keyboard.is_pressed('left ctrl'):
                print('Training data collection begin!')
                break_flag=0
                while not keyboard.is_pressed('left shift'):
                    label,EMG=serial_reader(ser)
                    EMG_storage,EMG_queue,label_storage,label_queue=shape_samples(EMG_storage,EMG_queue,EMG,label_storage,label_queue,label,sample_length=sample_length)
                    print(EMG_queue.shape,EMG_storage.shape)
                    print(f'label:{label}\ncollected data:{label_storage.shape[0]}')
                    '''plot_queue=np.concatenate((plot_queue,EMG),axis=1)
                    plt.clf()
                    for i in range(8):
                        plt.subplot(811+i)
                        plt.plot(plot_queue[i,-500:])
                        plt.ylim((-160,160))
                    plt.draw()
                    plt.pause(0.01)'''
                    if keyboard.is_pressed('esc'):
                        break_flag=1
                        break
                if break_flag:
                    break
                print(label_storage.shape,EMG_storage.shape)
                print('Training data collection finish!')
                print(f'{label_storage.shape[0]} data point is collected.')
                samples=(EMG_storage,label_storage)
                return samples
            else:
                label,EMG=serial_reader(ser)
                '''plot_queue=np.concatenate((plot_queue,EMG),axis=1)
                plt.clf()
                for i in range(8):
                    plt.subplot(811+i)
                    plt.plot(plot_queue[i,-500:])
                    if serial_reader==serial_reader_full:
                        plt.ylim((-160,160))
                    else:
                        plt.ylim((-2200,2200))
                plt.draw()
                plt.pause(0.01)'''
                if p>=100:
                    print(f'label:{label}\nEMG_data:{EMG.shape}')
                    p=0
                else:
                    p+=1
                if keyboard.is_pressed('esc'):
                    break
        print('Exit')
        return False
#短时傅里叶变换
def stft_transform(signal, fs, n_fft=256, hop_length=None):
    if hop_length is None:
        hop_length = n_fft // 4
    f, t, Zxx = stft(signal, fs=fs, nperseg=n_fft, noverlap=hop_length, nfft=n_fft)
    return f, t, np.abs(Zxx)
#连续小波变换
def cwt_transform(signal, fs, scales, wavelet='cmor'):
    coef, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    return coef
#转化为rgb图像
def create_rgb_image(stft_result, cwt_result, signals):
    n_channels = stft_result.shape[0]
    height = stft_result.shape[1]
    width = stft_result.shape[2]
    # Create an empty RGB image
    rgb_image = np.zeros((height, width, 3))
    # STFT for each channel
    for i in range(n_channels):
        stft_magnitude = np.abs(stft_result[i])
        # Normalize to [0, 255]
        stft_magnitude = (stft_magnitude - stft_magnitude.min()) / (stft_magnitude.max() - stft_magnitude.min()) * 255
        rgb_image[:, :, 0] += stft_magnitude[i, :, :]
    # CWT for each channel
    for i in range(n_channels):
        cwt_magnitude = np.abs(cwt_result[i])
        # Normalize to [0, 255]
        cwt_magnitude = (cwt_magnitude - cwt_magnitude.min()) / (cwt_magnitude.max() - cwt_magnitude.min()) * 255
        rgb_image[:, :, 1] += cwt_magnitude[i, :, :]
    # Use the original signal amplitude for the third channel
    for i in range(n_channels):
        signal_magnitude = np.abs(np.fft.fft(signals))
        signal_magnitude = (signal_magnitude - signal_magnitude.min()) / (signal_magnitude.max() - signal_magnitude.min()) * 255
        rgb_image[:, :, 2] += signal_magnitude[i]
    return rgb_image
#信号重构为RGB图像
def Transform_EMG(emg_signals,fs = 500,n_fft = 256,cwt_scales= np.arange(1, 128)):
    # STFT
    hop_length = n_fft // 4
    stft_results = np.array([stft_transform(emg_signal, fs, n_fft, hop_length)[2] for emg_signal in emg_signals.T])
    # CWT
    cwt_results = np.array([cwt_transform(emg_signal, fs, cwt_scales, wavelet='cmor') for emg_signal in emg_signals.T])
    # Convert to RGB image
    rgb_image = create_rgb_image(stft_results, cwt_results, emg_signals)
    return rgb_image
#数据预处理
def preparing(samples):
    X,y=samples
    label_set=set(y)
    label_dic={}
    for i in range(len(label_set)):
        label_dic[list(label_set)[i]]=i
    for j in range(len(y)):
        y[j]=label_dic[y[j]]
    y_categorical = to_categorical(y)  # one-hot 编码标签
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_dic
#模型构建
def CNN_model(input_shape, n_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    return model
#神经网络学习
def Learning(samples,save_path):
    #数据预处理
    trans_samples=(Transform_EMG(samples[0]),samples[1])
    X_train, X_test, y_train, y_test, label_dic=preparing(trans_samples)
    classes_num=len(label_dic)
    trans_dic={value: key for key, value in label_dic.items()}
    #构建模型
    input_shape = X_train.shape[1:]  # 图像的形状，不包括批次大小
    n_classes = y_train.shape[1]  # 类别数
    model = CNN_model(input_shape, n_classes)
    #编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 定义回调函数
    checkpointer = ModelCheckpoint(filepath='models/best_model.keras', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=10, verbose=1)
    # 训练模型
    history = model.fit(X_train, y_train, batch_size=16, epochs=100, verbose=1,
                        validation_split=0.2,  # 假设20%的数据用于验证
                        callbacks=[checkpointer, early_stopping])
    # 评估模型
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    # 使用classification_report来获取每个类别的准确率和其他指标
    report = classification_report(y_true, y_pred_classes, target_names=[f'Class {trans_dic[i]}' for i in range(classes_num)])
    print(report)
    # 可视化训练历史
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    return model,trans_dic
#模型保存
def save_model(save_path,model,translator):
    model_file='models/classifier_model.h5'
    model.save(model_file)
    json_str = json.dumps(translator)
    with open('models/translator.json', 'w') as f:
        f.write(json_str)
    print(f'Save model sucessfully!\nSaving path:{save_path}')
    return True
#模型加载
def load_model(save_path):
    model_file='models/classifier_model.h5'
    model = tf.keras.models.load_model(model_file)
    with open('models/translator.json', 'r') as f:
        translator = json.load(f)
    print('Load model sucessfully!')
    return model,translator
#获取分类函数
def get_classifier(model,trans_dic):
    def classifier(EMG_queue):
        X=np.array([EMG_queue])
        y_pred = model.predict(X,verbose=0)
        y_pred_classes = np.argmax(y_pred)%y_pred.shape[1]
        return trans_dic[str(y_pred_classes)]
    return classifier
#进行分类
def Classify(classifier,serial_reader,length=50):
    with serial.Serial(port="COM4", baudrate=115200, timeout=0.1) as ser:
        label_queue,EMG_queue=serial_reader(ser)
        print('Preparing finish. Press \'Space\' to classify.')
        while not keyboard.is_pressed('space'):
            pass
        print('Begin classify. Press \'Esc\' to exit.')
        while not keyboard.is_pressed('esc'):
            label_queue,EMG_queue=serial_reader(ser)
            for k in range(int(length/10)-1):
                label,EMG=serial_reader(ser)
                EMG_queue=np.concatenate((EMG_queue,EMG),axis=1)
            print(classifier(EMG_queue))
        print('Exit')
if __name__ == '__main__':
    SLECTION_DICT={'Exit':0,
                    'Collecting samples':1,
                    'Learning with saved samples':2,
                    'Learning with collected samples':3,
                    'Classify with saved model':4,
                    'Classify with new model':5
                    }
    selection=2
    save_path='models'
    samples_file=save_path+'\\Samples.npz'
    samples_length=500
    serial_reader=serial_reader_pure
    match selection:
        case 0:
            pass
        case 1:
            samples=Collect_samples(serial_reader,samples_length)
            if samples:
                save_samples(samples_file,samples)
            else:
                print('Fail to save samples.')
        case 2:
            samples=load_samples(samples_file)
            model,translator=Learning(samples,save_path)
            save_model(save_path,model,translator)
        case 3:
            samples=Collect_samples(serial_reader,samples_length)
            save_samples(samples_file,samples)
            model,translator=Learning(samples,save_path)
            save_model(save_path,model,translator)
        case 4:
            model,translator=load_model(save_path)
            classifier=get_classifier(model,translator)
            Classify(classifier,serial_reader,samples_length)
        case 5:
            samples=Collect_samples(serial_reader,samples_length)
            save_samples(samples_file,samples)
            model,translator=Learning(samples,save_path)
            classifier=get_classifier(model,translator)
            Classify(classifier,serial_reader,samples_length)