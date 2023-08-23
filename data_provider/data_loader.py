#将预处理数据转变为需要的序列数据

import numpy as np
import pandas as pd


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
output_feature：目标输出特征长度
path：数据文件地址
numbers：需要读取轴承编号（默认使用1工况轴承）
header：None第一行没有表头，0第一行为表头
'''
def read_data(features_size,output_feature,path,numbers,header):
    index_columns_names = ["UnitNumber", "RUL", "ScaRUL"]  ##RUL做完了后自己先进行归一化，这样更加方便训练预测

    features_colums = ['s' + str(i) for i in range(1, features_size+1)]
    input_file_column_names = index_columns_names + features_colums

    datas=[]
    for i in numbers:
        datas.append(pd.read_csv(path+f"bearing1_{i}.csv"),sep=',',names=input_file_column_names,header=header)

    return pd.concat(datas,axis=0)


