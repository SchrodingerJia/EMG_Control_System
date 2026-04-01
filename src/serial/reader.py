import numpy as np
import serial
import keyboard

from src.utils.helpers import get_label

def serial_reader_full(ser):
    """串口读取(全数据)"""
    ser.read_all()
    EMG_samples=[[] for i in range(8)]
    read_flag = 0
    while not read_flag:
        com_input = ser.read(1).hex()
        if com_input=='aa':
            com_input = ser.read(1).hex()
            if com_input=='aa':
                label=get_label()
                while ser.inWaiting()<96:
                    pass
                read_flag=1
    com_input = ser.read(ser.inWaiting())
    for package_index in range(10):
        for EMG_index in range(8):
            start_index=14+package_index*8+EMG_index
            EMG_samples[EMG_index].append([int(com_input[start_index:start_index+1].hex(),16)-127])
    return np.array([label]),np.array(EMG_samples)

def serial_reader_pure(ser):
    """串口读取(纯肌电)"""
    ser.readline()
    EMG=[[] for i in range(8)]
    label=get_label()
    while ser.inWaiting()<400:
        pass
    com_input = ser.read(ser.inWaiting())
    for package_index in range(10):
        data=str(com_input,encoding='utf-8').split('\n')
        for EMG_index in range(8):
            EMG[EMG_index].append([int(data[package_index].split(' ')[EMG_index])-2048])
    return np.array([label]),np.array(EMG)