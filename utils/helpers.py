import numpy as np
import keyboard

def get_label():
    """获取标签"""
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

def mode(array):
    """获得众数"""
    count = np.count_nonzero(array == 'IDLE')
    if count>=int(array.shape[0]):
        return 'IDLE'
    else:
        vals, counts = np.unique(array[array!='IDLE'], return_counts=True)
        index = np.argmax(counts)
        return vals[index]

def stablize(lst):
    """判断是否有连续三个稳定值"""
    array=np.array(lst)
    if np.all(array == array[0]):
        return lst[0]
    else:
        return None