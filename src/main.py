import numpy as np
import time
import serial
import serial.tools.list_ports
import keyboard
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, SimpleRNN, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, PReLU, LeakyReLU, AveragePooling2D, Flatten, Dense, Dropout, SpatialDropout2D, SpatialDropout1D  # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.constraints import max_norm # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
from ftplib import FTP
#串口读取(全数据)
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
#串口读取(纯肌电)
def serial_reader_pure(ser):
    ser.readline()
    EMG=[[] for i in range(8)]
    label=get_lable()
    while ser.inWaiting()<400:
        pass
    com_input = ser.read(ser.inWaiting())
    for package_index in range(10):
        data=str(com_input,encoding='utf-8').split('\n')
        for EMG_index in range(8):
            EMG[EMG_index].append([int(data[package_index].split(' ')[EMG_index])-2048])
    return np.array([label]),np.array(EMG)
#获取标签
def get_lable():
    label=''
    KEY_LIST=['q','w','e','r','t','y','u','i','o','p',
               'a','s','d','f','g','h','j','k','l',
                'z','x','c','v','b','n','m','1','2','3','4','5','6']
    for key in KEY_LIST:
        if keyboard.is_pressed(key):
            label=key
    if label:
        return label
    else:
        return 'IDLE'
#获得众数
def mode(array):
    count = np.count_nonzero(array == 'IDLE')
    if count>=int(array.shape[0]):
        return 'IDLE'
    else:
        vals, counts = np.unique(array[array!='IDLE'], return_counts=True)
        index = np.argmax(counts)
        return vals[index]
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
def Collect_samples(serial_reader):
    with serial.Serial(port="COM4", baudrate=115200, timeout=0.1) as ser:
        #样本存储初始化
        label_storage,EMG_storage=serial_reader(ser)
        print('Preparing finish.\nPress \'left Ctrl\' to start collecting samples.\nPress \'left Shift\' to stop collecting samples.')
        p=0
        while True:
            if keyboard.is_pressed('left ctrl'):
                print('Training data collection begin!')
                break_flag=0
                while not keyboard.is_pressed('left shift'):
                    label,EMG=serial_reader(ser)
                    label_storage=np.concatenate((label_storage,label),axis=0)
                    EMG_storage=np.concatenate((EMG_storage,EMG),axis=1)
                    print(f'label:{label}\ncollected data:{label_storage.shape[0]}')
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
                if p>=100:
                    print(f'label:{label}\nEMG_data:{EMG.shape}')
                    p=0
                else:
                    p+=1
                if keyboard.is_pressed('esc'):
                    break
        print('Exit')
        return False
#数据重构
def reshape_samples(samples,length=50):
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
#数据预处理
def preparing(samples,test_size=0.2):
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
#模型构建
def CNN_model(nb_classes, Chans=8, Samples=50, dropoutRate=0.4, kernLength=24, F1=12, D=1, F2=24, norm_rate=0.75, dropoutType = Dropout):
    input1 = Input(shape=(Chans, Samples, 1))
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((1, Chans), use_bias=False, depth_multiplier=D)(block1)
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU()(block1)
    block1 = AveragePooling2D((1, 2))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    block2 = Conv2D(F2, (1, 8), padding='same', use_bias=False)(block1)
    block2 = BatchNormalization()(block2)
    block2 = LeakyReLU()(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    block2 = dropoutType(dropoutRate)(block2)
    flatten = Flatten()(block2)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate), activation='softmax')(flatten)
    return Model(inputs=input1, outputs=dense)
def RNN_model(nb_classes, Chans=8, Samples=50, dropoutRate=0.7, units1=64, units2=32, norm_rate=0.25, dropoutType='Dropout'):
    # 确定dropout类型
    if dropoutType == 'SpatialDropout1D':
        dropoutType = SpatialDropout1D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout1D or Dropout, passed as a string.')
    # 定义输入层
    input_layer = Input(shape=(Chans, Samples))
    # 第一个RNN层
    rnn1 = SimpleRNN(units=units1, return_sequences=True)(input_layer)
    rnn1 = BatchNormalization()(rnn1)
    rnn1 = dropoutType(dropoutRate)(rnn1)
    # 第二个RNN层
    rnn2 = SimpleRNN(units=units2, return_sequences=False)(rnn1)
    rnn2 = BatchNormalization()(rnn2)
    rnn2 = dropoutType(dropoutRate)(rnn2)
    # 全连接层
    dense_layer = Dense(nb_classes, kernel_constraint=max_norm(norm_rate), activation='softmax')(rnn2)
    # 创建模型
    model = Model(inputs=input_layer, outputs=dense_layer)
    return model
#神经网络学习
def Learning(samples,model_type,save_path,sample_length):
    # 数据预处理
    samples=reshape_samples(samples,length=sample_length)
    X_train, X_test, y_train, y_test, label_dic=preparing(samples)
    classes_num=len(label_dic)
    trans_dic={value: key for key, value in label_dic.items()}
    # 构建模型
    model = model_type(nb_classes=classes_num)
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 定义回调函数
    checkpointer = ModelCheckpoint(filepath='models/best_model.keras', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=10, verbose=1)
    # 训练模型
    model.fit(X_train, y_train, batch_size=16, epochs=100, verbose=1,
                        validation_split=0.2,  # 假设20%的数据用于验证
                        callbacks=[checkpointer, early_stopping])
    # 评估模型
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    #使用classification_report来获取每个类别的准确率和其他指标
    report = classification_report(y_true, y_pred_classes, target_names=[f'Class {trans_dic[i]}' for i in range(classes_num)])
    print(report)
    return model,trans_dic
#模型保存
def save_model(save_path,model,translator):
    model_file='models/classifier_model.keras'
    model.save(model_file)
    json_str = json.dumps(translator)
    with open('models/translator.json', 'w') as f:
        f.write(json_str)
    print(f'Save model sucessfully!\nSaving path:{save_path}')
    return True
#模型加载
def load_model(save_path):
    model_file='models/classifier_model.keras'
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
#判断是否有连续三个稳定值
def stablize(lst):
    # 遍历列表，检查是否有都是相同的值
    array=np.array(lst)
    if np.all(array == array[0]):
        return lst[0]
    else:
        return None
#进行分类
def Classify(classifier,serial_reader,path,length=50):
    # FTP服务器信息
    ftp_server = 'YOUR_FTP_SERVER_IP'
    ftp_user = 'YOUR_FTP_USERNAME'
    ftp_pass = 'YOUR_FTP_PASSWORD'
    # 要上传的文件路径
    remote_file_path = '/FTP/test.txt'  # 目标路径，根据需要修改
    # 创建FTP对象并连接
    ftp = FTP(ftp_server)
    ftp.login(user=ftp_user, passwd=ftp_pass)
    with serial.Serial(port="COM4", baudrate=115200, timeout=0.1) as ser:
        label,EMG_queue=serial_reader(ser)
        label_queue=[]
        p=0
        print('Preparing finish. Press \'Space\' to classify.')
        while not keyboard.is_pressed('space'):
            pass
        print('Begin classify. Press \'Esc\' to exit.')
        while not keyboard.is_pressed('esc'):
            label,EMG=serial_reader(ser)
            if EMG_queue.shape[1] >= length:
                if len(label_queue)==3:
                    stabel_label=stablize(label_queue)
                    if stabel_label!= None and stabel_label!='IDLE' and p>10:
                        with open('data/control_signal.txt', 'w') as f:
                            f.write(stabel_label)
                        print(f'Model value "{stabel_label}" has been written to data/control_signal.txt')
                        with open('data/control_signal.txt', 'rb') as file:
                            ftp.storbinary(f'STOR {remote_file_path}', file)
                        label_queue=[]
                        p=0
                    label_queue.append(classifier(EMG_queue))
                    label_queue=label_queue[-3:]
                    p+=1
                else:
                    label_queue.append(classifier(EMG_queue))
                print(label_queue)
                EMG_queue=np.concatenate((EMG_queue,EMG),axis=1)
                EMG_queue=EMG_queue[:,-length:]
            else:
                EMG_queue=np.concatenate((EMG_queue,EMG),axis=1)
        ftp.quit()
        print('Exit')
if __name__ == '__main__':
    SLECTION_DICT={'Exit':'esc',
                    'Collecting samples':'C',
                    'Adding samples':'A',
                    'Learning with saved samples':'LS',
                    'Learning with collected samples':'LC',
                    'Classify with saved model':'CS',
                    'Classify with new model':'CC'
                    }
    selection='LS'
    save_path='data\\raw'
    samples_file=save_path+'\\Samples.npz'
    sample_length=50
    serial_reader=serial_reader_pure
    model_type=CNN_model
    match selection:
        case 'esc':
            pass
        case 'C':
            samples=Collect_samples(serial_reader)
            if samples:
                save_samples(samples_file,samples)
            else:
                print('Fail to save samples.')
        case 'A':
            samples=Collect_samples(serial_reader)
            if samples:
                old_samples=load_samples(samples_file)
                all_EMGs=np.concatenate((old_samples[0],samples[0]),axis=1)
                all_labels=np.concatenate((old_samples[1],samples[1]),axis=0)
                print(f'{all_labels.shape[0]} old samples are loaded.')
                samples=(all_EMGs,all_labels)
                save_samples(samples_file,samples)
            else:
                print('Fail to save samples.')
        case 'LS':
            samples=load_samples(samples_file)
            model,translator=Learning(samples,model_type,save_path,sample_length)
            save_model(save_path,model,translator)
        case 'LC':
            samples=Collect_samples(serial_reader)
            save_samples(samples_file,samples)
            model,translator=Learning(samples,model_type,save_path,sample_length)
            save_model(save_path,model,translator)
        case 'CS':
            model,translator=load_model(save_path)
            classifier=get_classifier(model,translator)
            Classify(classifier,serial_reader,save_path,sample_length)
        case 'CC':
            samples=Collect_samples(serial_reader)
            save_samples(samples_file,samples)
            model,translator=Learning(samples,model_type,save_path,sample_length)
            save_model(save_path,model,translator)
            model,translator=load_model(save_path)
            classifier=get_classifier(model,translator)
            Classify(classifier,serial_reader,save_path,sample_length)
        case 'LCS':
            samples=load_samples(samples_file)
            model,translator=Learning(samples,model_type,save_path,sample_length)
            save_model(save_path,model,translator)
            model,translator=load_model(save_path)
            classifier=get_classifier(model,translator)
            Classify(classifier,serial_reader,save_path,sample_length)
        case _:
            samples=load_samples(samples_file)
            print(samples[0].shape)
            samples=reshape_samples(samples,length=sample_length)
            print(samples[0].shape)