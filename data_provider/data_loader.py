#将预处理数据转变为需要的序列数据

import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('.')
from torch.utils.data import DataLoader,TensorDataset
from data_provider.data_preprocess import scarle_rul
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
min_max_scaler=MinMaxScaler(feature_range=(0,1))
max_abs_scaler=MaxAbsScaler()

#将输入序列数据通过滑动窗口制作时间序列样本
def gen_test(id_df, seq_length, seq_cols, mask_value):
    df_mask = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    df_mask[:] = mask_value

    id_df = df_mask.append(id_df, ignore_index=True)

    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []

    for start, stop in zip(range(0, num_elements - seq_length + 1), range(seq_length, num_elements + 1)):
        lstm_array.append(data_array[start:stop, :])

    return np.array(lstm_array)

'''
读取预处理多个轴承数据并进行拼接,输出为DataFrame
feature_size：需要读取数据原始特征数据长度
path：数据文件地址
numbers：需要读取轴承编号（默认使用1工况轴承）
header：None第一行没有表头，0第一行为表头
'''
def read_data(features_size,path,numbers,header,degradation_ratio):
    index_columns_names = ["UnitNumber", "RUL", "ScaRUL"]  ##RUL做完了后自己先进行归一化，这样更加方便训练预测

    features_colums = ['s' + str(i) for i in range(1, features_size+1)]
    input_file_column_names = index_columns_names + features_colums

    datas=[]
    for i in numbers:
        data=pd.read_csv(path+f"bearing1_{i}.csv",sep=',',names=input_file_column_names,header=header)
        data["ScaRUL"]=scarle_rul(data,degradation_ratio)
        datas.append(data)

    return pd.concat(datas,axis=0)

'''
返回序列train/val/test三个数据迭代器
feature_size：需要读取数据原始特征数据长度
path：数据文件地址
header：None第一行没有表头，0第一行为表头
output_feature：目标输出特征长度
sequence_length：需要生成的序列长度
degradation_ratio：退化数据比例
'''
def sequence_data_loader(features_size,path,header,output_feature_size,sequence_length,degradation_ratio):
    mask_value=0#没有掩码操作
    feats=['s'+str(i) for i in range(1,output_feature_size+1)]#需要使用的特征

#获取数据
    train_df=read_data(features_size,path,[1,2,3],header,degradation_ratio)#使用1,2,3号轴承作为训练集
    val_df=read_data(features_size,path,[5],header,degradation_ratio)#使用5号轴承作为训练集
    test_df = read_data(features_size, path, [6,7], header,degradation_ratio)  # 使用5号轴承作为训练集

#数据归一化
    train_df[feats]=max_abs_scaler.fit_transform(train_df[feats])
    val_df[feats]=max_abs_scaler.transform(val_df[feats])
    test_df[feats] = max_abs_scaler.transform(test_df[feats])

 # 生成带时间步的三维数据，前面添加空行，滑动步长为1，制作原始样本长度各序列样本
    x_train = np.concatenate(
        list(list(gen_test(train_df[train_df['UnitNumber'] == unit], sequence_length, feats, mask_value))
             for unit in train_df['UnitNumber'].unique()))  # unique()获取唯一值
    print("x_train shape:",x_train.shape)
    x_val = np.concatenate(list(list(gen_test(val_df[val_df['UnitNumber'] == unit], sequence_length, feats, mask_value))
                                for unit in val_df['UnitNumber'].unique()))
    print("x_test shape:",x_val.shape)
    x_test = np.concatenate(
        list(list(gen_test(test_df[test_df['UnitNumber'] == unit], sequence_length, feats, mask_value))
             for unit in test_df['UnitNumber'].unique()))
    print(x_test.shape)

#获取数据RUL标签
    y_train = train_df.ScaRUL.values  ###这里的ScaRUL是列名，如果不是这样命名要修改
    y_val = val_df.ScaRUL.values
    y_test = test_df.ScaRUL.values

#转化为tensor数据类型
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_val=torch.Tensor(x_val)
    y_val=torch.Tensor(y_val)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    print("训练集的特征形状, x_train ：", x_train.shape)
    print("训练集的标签形状, y_train ：", y_train.shape)
    print("验证集的特征性状，x_val ：", x_val.shape)
    print("验证集的标签形状，y_val ：", y_val.shape)
    print("测试集的特征形状, x_test ：", x_test.shape)
    print("测试集的标签形状, y_test ：", y_test.shape)

    data_train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=False,
                                   num_workers=0)
    data_val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=128, shuffle=False, num_workers=0)

    data_test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=False,
                                  batch_size=128, num_workers=0)

    return data_train_loader,data_val_loader,data_test_loader