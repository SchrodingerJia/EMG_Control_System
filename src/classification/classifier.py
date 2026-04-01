import numpy as np
import json
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report

from src.data.preprocessing import reshape_samples, preparing
from src.models.cnn_model import CNN_model

def Learning(samples, model_type, save_path, sample_length):
    """神经网络学习"""
    # 数据预处理
    samples=reshape_samples(samples, length=sample_length)
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
    
    return model, trans_dic

def save_model(save_path, model, translator):
    """模型保存"""
    model_file='models/classifier_model.keras'
    model.save(model_file)
    json_str = json.dumps(translator)
    with open('models/translator.json', 'w') as f:
        f.write(json_str)
    print(f'Save model sucessfully!\nSaving path:{save_path}')
    return True

def load_model(save_path):
    """模型加载"""
    model_file='models/classifier_model.keras'
    model = tf.keras.models.load_model(model_file)
    with open('models/translator.json', 'r') as f:
        translator = json.load(f)
    print('Load model sucessfully!')
    return model, translator

def get_classifier(model, trans_dic):
    """获取分类函数"""
    def classifier(EMG_queue):
        X=np.array([EMG_queue])
        y_pred = model.predict(X, verbose=0)
        y_pred_classes = np.argmax(y_pred)%y_pred.shape[1]
        return trans_dic[str(y_pred_classes)]
    return classifier