import numpy as np
import pandas as pd
import os
from scipy import stats
import pywt


#通过滑动窗口制作样本及标签
def data_slide_window():
    N=2560#窗口长度2560
    degradation_ratio=0.8#退化阶段数据占比
    condition_num=str(1)
    signal_hor_ver='hor'
    #原始2560维振动信号
    save_pth=f"PHM2012/原始2560维振动数据/退化比例{degradation_ratio}/"
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    for bearing_num in range(1,8):
        read_path='PHM2012/原始数据一维拼接/'+condition_num+"_{}_".format(bearing_num)+signal_hor_ver+'.npy'
        bearing_data=np.load(read_path)
        bearing_data=bearing_data.reshape(-1,2560)
        max_rul=int(bearing_data.shape[0]*degradation_ratio)#获得进入退化阶段最大样本数
        other_parameters=[[bearing_num,max_rul,1]]*(bearing_data.shape[0]-max_rul)
        #获得轴承标号及剩余使用寿命
        for i in range(max_rul):
            other_parameters.append([bearing_num,max_rul-i-1,(max_rul-i-1)/(max_rul)])
        other_parameters=np.array(other_parameters)
        bearing_data=np.hstack((other_parameters, bearing_data))
        np.savetxt(save_pth+"bearing1_" + str(bearing_num) + ".csv", bearing_data, fmt='%f',delimiter=',')
data_slide_window()
'''
标签制作
data：输入数据DataFrame
degradation_ration：退化比例
'''
def scarle_rul(data,degradation_ratio):
    max_rul = int(data.shape[0] * degradation_ratio)  # 获得进入退化阶段最大样本数
    rul=[1]*(data.shape[0]-max_rul)
    for i in range(max_rul):
        rul.append((max_rul - i - 1) / (max_rul))

    return  rul


'''
提取phm2012时域和频域特征
'''
#时域特征提取
def get_time_domain_feature(data):
    """
    提取 15个 时域特征

    @param data: shape 为 (m, n) 的 2D array 数据，其中，m 为样本个数， n 为样本（信号）长度
    @return: shape 为 (m, 15)  的 2D array 数据，其中，m 为样本个数。即 每个样本的16个时域特征
    """
    rows, cols = data.shape

    # 有量纲统计量
    max_value = np.amax(data, axis=1)  # 最大值
    peak_value = np.amax(abs(data), axis=1)  # 最大绝对值
    min_value = np.amin(data, axis=1)  # 最小值
    mean = np.mean(data, axis=1)  # 均值
    p_p_value = max_value - min_value  # 峰峰值
    abs_mean = np.mean(abs(data), axis=1)  # 绝对平均值
    rms = np.sqrt(np.sum(data ** 2, axis=1) / cols)  # 均方根值
    square_root_amplitude = (np.sum(np.sqrt(abs(data)), axis=1) / cols) ** 2  # 方根幅值
    # variance = np.var(data, axis=1)  # 方差
    std = np.std(data, axis=1)  # 标准差
    kurtosis = stats.kurtosis(data, axis=1)  # 峭度
    skewness = stats.skew(data, axis=1)  # 偏度
    # mean_amplitude = np.sum(np.abs(data), axis=1) / cols  # 平均幅值 == 绝对平均值

    # 无量纲统计量
    clearance_factor = peak_value / square_root_amplitude  # 裕度指标
    shape_factor = rms / abs_mean  # 波形指标
    impulse_factor = peak_value / abs_mean  # 脉冲指标
    crest_factor = peak_value / rms  # 峰值指标
    # kurtosis_factor = kurtosis / (rms**4)  # 峭度指标

    features = [max_value, peak_value, min_value, mean, p_p_value, abs_mean, rms, square_root_amplitude,
                std, kurtosis, skewness, clearance_factor, shape_factor, impulse_factor, crest_factor]

    return np.array(features).T

#频域特征
def get_frequency_domain_feature(data, sampling_frequency):
    """
    提取 4个 频域特征

    @param data: shape 为 (m, n) 的 2D array 数据，其中，m 为样本个数， n 为样本（信号）长度
    @param sampling_frequency: 采样频率，phm2012为256000
    @return: shape 为 (m, 4)  的 2D array 数据，其中，m 为样本个数。即 每个样本的4个频域特征
    """
    data_fft = np.fft.fft(data, axis=1)
    m, N = data_fft.shape  # 样本个数 和 信号长度

    # 傅里叶变换是对称的，只需取前半部分数据，否则由于 频率序列 是 正负对称的，会导致计算 重心频率求和 等时正负抵消
    mag = np.abs(data_fft)[:, : N // 2]  # 信号幅值
    freq = np.fft.fftfreq(N, 1 / sampling_frequency)[: N // 2]
    # mag = np.abs(data_fft)[: , N // 2: ]  # 信号幅值
    # freq = np.fft.fftfreq(N, 1 / sampling_frequency)[N // 2: ]

    ps = mag ** 2 / N  # 功率谱

    fc = np.sum(freq * ps, axis=1) / np.sum(ps, axis=1)  # 重心频率
    mf = np.mean(ps, axis=1)  # 平均频率
    rmsf = np.sqrt(np.sum(ps * np.square(freq), axis=1) / np.sum(ps, axis=1))  # 均方根频率

    freq_tile = np.tile(freq.reshape(1, -1), (m, 1))  # 复制 m 行
    fc_tile = np.tile(fc.reshape(-1, 1), (1, freq_tile.shape[1]))  # 复制 列，与 freq_tile 的列数对应
    vf = np.sum(np.square(freq_tile - fc_tile) * ps, axis=1) / np.sum(ps, axis=1)  # 频率方差


    features = [fc, mf, rmsf, vf]

    return np.array(features).T

#时频域特征
def get_wavelet_packet_feature(data, wavelet='db3', mode='symmetric', maxlevel=3):
    """
    提取 小波包特征

    @param data: shape 为 (n, ) 的 1D array 数据，其中，n 为样本（信号）长度
    @return: 最后一层 子频带 的 能量百分比
    """
    wp = pywt.WaveletPacket(data, wavelet=wavelet, mode=mode, maxlevel=maxlevel)

    nodes = [node.path for node in wp.get_level(maxlevel, 'natural')]  # 获得最后一层的节点路径

    e_i_list = []  # 节点能量
    for node in nodes:
        e_i = np.linalg.norm(wp[node].data, ord=None) ** 2  # 求 2范数，再开平方，得到 频段的能量（能量=信号的平方和）
        e_i_list.append(e_i)

    # 以 频段 能量 作为特征向量
    # features = e_i_list

    # 以 能量百分比 作为特征向量，能量值有时算出来会比较大，因而通过计算能量百分比将其进行缩放至 0~100 之间
    e_total = np.sum(e_i_list)  # 总能量
    features = []
    for e_i in e_i_list:
        features.append(e_i / e_total * 100)  # 能量百分比

    return np.array(features)


def feature_calculation(input_path,output_path):
    index_columns_names = ["UnitNumber", "RUL", "ScaRUL"]  ##RUL做完了后自己先进行归一化，这样更加方便训练预测

    features_colums = ['s' + str(i) for i in range(1, 2560 + 1)]
    input_file_column_names = index_columns_names + features_colums
    #一工况有7个轴承，4数据有问题直接跳过
    for i in range(1,8):
        if i==4:
            output=np.empty((2, 2))
            np.savetxt(output_path+f'bearing1_{i}.csv',output,delimiter=',')
            continue

        output_data=pd.DataFrame()
        read_data=pd.read_csv(input_path+f"bearing1_{i}.csv",sep=",",header=None,names=input_file_column_names)
        raw_data=read_data[features_colums]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        #时域指标
        time_features=get_time_domain_feature(raw_data.values)

        #频域指标
        frequence_features=get_frequency_domain_feature(raw_data,256000)

        #时频域特征
        time_frequency_doamin_frequency = []

        # 通过for循环每次提取一个样本的时-频域特征
        for raw in range(raw_data.shape[0]):
            wavelet_packet_feature = get_wavelet_packet_feature(raw_data.values[raw])
            time_frequency_doamin_frequency.append(wavelet_packet_feature)

        time_frequency_doamin_frequency = np.array(time_frequency_doamin_frequency)

        features=np.concatenate([time_features,frequence_features,time_frequency_doamin_frequency],axis=1)
        output=np.concatenate([read_data.values[:,0:3],features],axis=1)
        np.savetxt(output_path+f'bearing1_{i}.csv',output,delimiter=',',fmt='%.6f')
        print(f"bearing1_{i}已完成特征提取")


