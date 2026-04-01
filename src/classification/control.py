import numpy as np
import serial
import keyboard
from ftplib import FTP

from src.utils.helpers import stablize

def Classify(classifier, serial_reader, path, length=50):
    """进行分类"""
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