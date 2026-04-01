from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, LeakyReLU, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.constraints import max_norm

def CNN_model(nb_classes, Chans=8, Samples=50, dropoutRate=0.4, kernLength=24, F1=12, D=1, F2=24, norm_rate=0.75, dropoutType=Dropout):
    """构建CNN模型"""
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