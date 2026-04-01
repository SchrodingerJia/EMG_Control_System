# 可穿戴肌电臂环实时控制外部设备系统

## 项目简介
本项目是一个基于可穿戴肌电臂环的机械臂控制系统。通过采集人体手臂的肌电信号，经过信号处理和模式识别，实现对机械臂的实时控制。项目采用模块化设计，包含信号采集、处理、模型训练和控制等多个环节。

## 项目结构
```
code/
├── README.md               # 项目说明文档
├── requirements.txt        # 项目依赖包列表
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   │   └── Samples.npz    # 采集的肌电信号样本
│   └── control_signal.txt # 控制信号输出文件
├── models/                 # 模型目录
│   ├── best_model.keras   # 训练过程中保存的最佳模型
│   ├── classifier_model.h5 # 图像识别模型（备用）
│   ├── classifier_model.keras # 主要分类模型
│   └── translator.json    # 标签映射文件
├── src/                    # 源代码目录
│   ├── main.py            # 主程序入口
│   ├── data/              # 数据处理模块
│   │   ├── __init__.py
│   │   ├── collection.py  # 数据采集功能
│   │   └── preprocessing.py # 数据预处理功能
│   ├── models/            # 模型定义模块
│   │   ├── __init__.py
│   │   ├── cnn_model.py   # CNN模型定义
│   │   └── rnn_model.py   # RNN模型定义
│   ├── features/          # 特征工程模块
│   │   ├── __init__.py
│   │   └── feature_extraction.py # 特征提取函数
│   ├── serial/            # 串口通信模块
│   │   ├── __init__.py
│   │   └── reader.py      # 串口数据读取
│   ├── classification/    # 分类控制模块
│   │   ├── __init__.py
│   │   ├── classifier.py  # 分类器功能
│   │   └── control.py     # 控制逻辑
│   └── experiments/       # 实验代码
│       └── 001_stft_cwt_cnn.py # 时频图CNN实验
└── utils/                 # 工具函数
    ├── __init__.py
    ├── helpers.py         # 辅助函数
    └── ftp_client.py      # FTP客户端
```

## 功能模块

### 1. 肌电信号采集与传输
- 通过串口实时采集8通道肌电信号
- 支持两种数据格式：完整数据包和纯肌电数据
- 使用键盘输入作为动作标签

### 2. 信号处理与模式识别
项目探索了三种实现路径：
1. **特征提取+SVM分类**：提取时域和频域特征，使用传统机器学习方法
2. **时频图+图像识别模型**：将信号转换为STFT和CWT时频图，构建RGB图像，使用CNN进行分类
3. **CNN神经网络模型**：直接对原始信号进行卷积神经网络处理（最终采用方案）

### 3. 机械臂通信与控制
- 通过FTP协议将控制信号传输到机械臂控制器
- 实现稳定检测机制，确保控制指令的可靠性
- 支持实时分类和批量处理两种模式

## 使用方法

### 环境配置
```bash
pip install -r requirements.txt
```

### 数据采集
1. 连接肌电臂环到COM4端口
2. 运行主程序：`python src/main.py`
3. 选择数据采集模式，按照提示操作

### 模型训练
1. 准备足够的训练数据
2. 选择训练模式，程序将自动进行数据预处理、模型训练和评估
3. 训练好的模型将保存在`models/`目录下

### 实时控制
1. 加载训练好的模型
2. 启动分类模式，程序将实时处理肌电信号并生成控制指令
3. 控制信号将通过FTP传输到机械臂

## 依赖包
主要依赖包包括：
- numpy
- tensorflow
- scipy
- scikit-learn
- pyserial
- pywt
- matplotlib
- ftplib

详细依赖见`requirements.txt`文件。

## 注意事项
1. 确保串口配置正确（默认COM4, 115200波特率）
2. FTP服务器配置需要根据实际环境修改
3. 数据采集时需要保持动作标准性和一致性
4. 模型训练需要足够的数据量和计算资源

## 许可证
本项目仅供学习研究使用。