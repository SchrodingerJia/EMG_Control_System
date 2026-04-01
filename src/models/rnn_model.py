from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, BatchNormalization, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.constraints import max_norm

def RNN_model(nb_classes, Chans=8, Samples=50, dropoutRate=0.7, units1=64, units2=32, norm_rate=0.25, dropoutType='Dropout'):
    """构建RNN模型"""
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