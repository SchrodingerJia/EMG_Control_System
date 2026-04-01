import numpy as np
import serial
import keyboard

from src.serial.reader import serial_reader_pure, serial_reader_full
from src.utils.helpers import get_label, mode

def save_samples(filename, samples):
    """保存EMG数据和标签"""
    np.savez(filename, EMG=samples[0], labels=samples[1])
    print(f'Save datas sucessfully!\nSaving file:{filename}')
    return True

def load_samples(filename):
    """加载EMG数据和标签"""
    EMG=np.load(filename)['EMG']
    labels=np.load(filename)['labels']
    print('Load datas sucessfully!')
    samples=(EMG,labels)
    return samples

def Collect_samples(serial_reader):
    """收集肌电信号样本"""
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