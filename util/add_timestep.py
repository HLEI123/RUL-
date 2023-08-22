import numpy as np
import pandas as pd 

def gen_test(id_df, seq_length, seq_cols, mask_value):

    df_mask = pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    df_mask[:] = mask_value
    
    id_df = df_mask.append(id_df,ignore_index=True)
    
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)
    
#自定义评价指标
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred -y_true)))

#自定义PHM2008评价函数
def Scoring_2008(Y_true, Y_pred):
    h = Y_pred - Y_true
    g = (-(h-np.abs(h))/2.0)
    f = ((np.abs(h)+h)/2.0)
    return np.sum(np.exp(g/13.0)-1)+np.sum(np.exp(f/10.0)-1)
    
# #PHM2012得分函数，未进行平均处理
# def Scoring2(Y_true, Y_pred):
    # h = (Y_true - Y_pred)/Y_true
    # g = (-(h-np.abs(h))/2.0)
    # f = ((np.abs(h)+h)/2.0)
    # return np.sum(np.exp(g/5.0))+np.sum(np.exp(f/20.0))
#PHM2012得分函数，进行平均

def Scoring2012(Y_true, Y_pred):
    h = (Y_true - Y_pred)/Y_true
    g = (-(h-np.abs(h))/2.0)
    f = ((np.abs(h)+h)/2.0)
    return (np.sum(np.exp(np.log(0.5)*(g/5.0)))+np.sum(np.exp(np.log(0.5)*(f/20.0))))/len(Y_true)