import numpy as np
from scipy.signal import welch

fs=500
window_lenth=128
#MF:中值频率
def MF(signal):
    #求功率谱密度
    freqs, psd = welch(signal, fs, nperseg=window_lenth)

    # 计算累积功率分布
    cumulative_power = np.cumsum(psd)
    
    # 找到累积功率达到50%的频率
    total_power = cumulative_power[-1]
    median_index = np.where(cumulative_power >= total_power / 2)[0][0]
    
    # 返回中值频率
    return freqs[median_index]

#MPF:均值频率
def MPF(signal):
    # 计算窗口数量
    num_windows = len(signal) // window_lenth
    
    # 初始化频率和功率的总和
    total_power = 0
    total_freq = 0
    
    # 遍历每个窗口
    for i in range(num_windows):
        # 提取当前窗口的信号
        window_signal = signal[i * window_lenth:(i + 1) * window_lenth]
        
        # 使用Welch方法估计功率谱密度
        freqs, psd = welch(window_signal, fs, nperseg=window_lenth)
        
        # 计算总功率
        total_power += np.sum(psd)
        
        # 计算频率的加权和
        total_freq += np.sum(freqs * psd)
    
    # 计算平均功率频率
    return total_freq / total_power

#IEMG:积分值
def IEMG(signal):
    return np.trapz(signal, dx=1)
#RMS:方均根值
def RMS(signal):
    return np.sqrt(np.mean(np.square(signal)))
#MAV:平均绝对值
def MAV(signal):
    return np.mean(np.absolute(signal))
#ZC:过零率
def ZC(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings)
#WL:波形长度
def WL(signal):
    return np.sum(np.absolute(np.diff(signal)))/len(signal)
#SSC:斜率符号变化
def SSC(signal):
    dead_zone=10e-7
    feature = 0
    for j in range(2, len(signal)):
        difference1 = signal[j-1] - signal[j-2]
        difference2 = signal[j-1] - signal[j]
        sign = difference1 * difference2
        
        if sign > 0:
            if abs(difference1) > dead_zone or abs(difference2) > dead_zone:
                feature += 1
    
    feature = feature/len(signal)
    return feature

def feature(signal):
    feature=[]
    #时域特征
    feature.append(IEMG(signal))
    feature.append(RMS(signal))
    feature.append(MAV(signal))
    feature.append(ZC(signal))
    feature.append(WL(signal))
    feature.append(SSC(signal))
    #频域特征
    feature.append(MF(signal))
    feature.append(MPF(signal))
    return feature
