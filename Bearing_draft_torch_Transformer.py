
import os
import sys
import math
from util.add_timestep import gen_test, Scoring_2008
from util.tcn import TemporalConvNet
# from util.model_torch import Gating,Encoder,EncoderLayer,MultiHeadAttention

import torch.nn.functional as F
from torch.nn.modules import MSELoss
import torch
from torch import nn
from torch import Tensor
from torch import optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('seaborn-whitegrid')#绘图的主题
plt.rcParams['font.sans-serif'] = 'Simsun'
plt.rcParams.update({'font.size': 12}) # 改变所有字体大小，改变其他性质类似

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# -------------------------------
# 读取训练集
# -------------------------------
####制作表头
"""这个需要你自己再csv的数据中提前把与处理好的FFT或者2560维度的原始振动数据中手动添加UnitNumber和RUL"""
index_columns_names =  ["UnitNumber","RUL","ScaRUL"]##RUL做完了后自己先进行归一化，这样更加方便训练预测


features_colums = ['s' + str(i) for i in range(1, 67)]###看你的数据维度，比如2560维度，那“67”就要改成2561
input_file_column_names = index_columns_names + features_colums

####归一的函数
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0,1))
max_ab_scaler = MaxAbsScaler()

###这个是数据集的根目录，到时候你自己修改，我直接把
path_dir = "../赠送/PHM2012_工况一_测试集全寿命_EMD_IMF_data/train_cond1/"

bearing1_1 = pd.read_csv(path_dir+'EMD_Bearing1_1.csv', sep=",",
                       names=index_columns_names+features_colums,header=0) #'FFT_1_1_hor.csv'
                    #header=None表示第一行没有表头；=0表示第一行为表头
bearing1_1

bearing1_2 = pd.read_csv(path_dir+'EMD_Bearing1_2.csv', sep=",",
                       names=index_columns_names+features_colums,header=0)
                    #header=None表示第一行没有表头；=0表示第一行为表头
bearing1_2

bearing1_3 = pd.read_csv(path_dir+'EMD_Bearing1_3.csv', sep=",",
                       names=index_columns_names+features_colums,header=0)
                    #header=None表示第一行没有表头；=0表示第一行为表头
bearing1_3


# ----------------------
# 数据集制作
# ----------------------
###要训练的轴承
train_df = pd.concat([bearing1_1,bearing1_2], axis=0)
train_df

###要测试的轴承
test_df = bearing1_3
test_df

###时间步
sequence_length = 64
mask_value = 0
feats =features_colums
feats = ['s' + str(i) for i in range(1, 67)]##需要的特征数据维度，可以自己选择

### 数据集归一化
"""先使用fit_transform,后使用transform"""

train_df[feats] = max_ab_scaler.fit_transform(train_df[feats])
test_df[feats] = max_ab_scaler.transform(test_df[feats])

#生成带时间步的三维数据
x_train=np.concatenate(list(list(gen_test(train_df[train_df['UnitNumber']==unit], sequence_length, feats, mask_value))
                           for unit in train_df['UnitNumber'].unique()))#unique()获取唯一值
print(x_train.shape)

#t获取训练集RUL标签
y_train = train_df.ScaRUL.values###这里的RUL是列名，如果不是这样命名要修改
y_train.shape

#生成带时间步的三维数据
x_test=np.concatenate(list(list(gen_test(test_df[test_df['UnitNumber']==unit], sequence_length, feats, mask_value))
                           for unit in test_df['UnitNumber'].unique()))
print(x_test.shape)

#t获取训练集RUL标签
y_test = test_df.ScaRUL.values
y_test.shape

##需要验证集，请自己从x_train中随机划分0.1-0.3的样本作为验证集
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
#x_vaild
#y_vaild
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

"""批大小提前在数据划分中自己定好，这是Torch中比较特殊的"""
data_train_loader  = DataLoader(TensorDataset(x_train,y_train),shuffle=False,
                               batch_size=128,num_workers=0)
data_test_loader = DataLoader(TensorDataset(x_test,y_test),shuffle=False,
                             batch_size=128,num_workers=0)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


"""这个就是transformer中的self attention的使用，这里是配合transformer中的Encoder一起使用"""


class Transforemer_Encoder(nn.Module):
    def __init__(self, feature_size, nhead, num_layers, dropout):
        super(Transforemer_Encoder, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.deco_output = nn.Linear(1, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        #         #我觉得可以不要位置编码
        #         src = self.pos_encoder(src)
        output = self.transformer_encoder(src, None)  # , self.src_mask)
        output = self.decoder(output)
        output = self.deco_output(output[:, -1, :])
        return F.sigmoid(output)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


"""在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。
    这是model中含有BN层和Dropout所带来的的性质。
    eval()时，框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，
    不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！
    在做one classification的时候，训练集和测试集的样本分布是不一样的，尤其需要注意这一点。
"""


##output.detach().cpu().numpy())#能缩短计算时间
def Epoch_evaluate(net, loader):
    net.eval()
    min_val_loss = 0.0
    val_rmse = 0.0
    val_score = 0.0
    i = 0
    preds = []
    true = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            #             data = data.view(-1, input_channels, seq_length)#permute(0,2,1)#
            # 上述一步是Torch中卷积操作独有的，如果是LSTM，transformer可以注释掉这行代码

            output = net(data)
            output = output.squeeze(1)

            preds.append(output.detach().cpu().numpy())
            true.append(target.detach().cpu().numpy())

            min_val_loss += Losses(output, target).item()
            i += 1
        min_val_loss = min_val_loss / i
        val_rmse = np.sqrt(min_val_loss)
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        #         val_score =np.float(Scoring_2008(true,preds))
        return round(min_val_loss, 4), np.round(val_rmse, 4)


# 现在都是对的了
def Last_evaluate2(net, loader):
    net.eval()
    min_val_loss = 0.0
    np_mse = 0.0
    preds = []
    true = []
    i = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            #             data = data.view(-1, input_channels, seq_length)#permute(0,2,1)
            # 上述一步是Torch中卷积操作独有的，如果是LSTM，transformer可以注释掉这行代码
            output = net(data)
            output = output.squeeze(1)
            preds.append(output.detach().cpu().numpy())
            true.append(target.detach().cpu().numpy())
            min_val_loss += Losses(output, target).item()
            i += 1
        min_val_loss = min_val_loss / i
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        return np.round(preds, 4), np.round(true, 2)

input_channels=x_train.shape[-1]
seq_length =sequence_length
epochs = 50
steps = 0

###设置transformer的基本参数
feature_size = input_channels  # 输入特征维度
nhead = 2  #  multi-head attention 的头数
num_layers = 1  # encoder layers 的层数
dropout = 0.25

# define and load model
model = Transforemer_Encoder(feature_size, nhead, num_layers, dropout).to(device)

# initialize Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=50,
                                                             min_lr=0.0001)###设置学习率下降策略，连续50次loss没有下降，学习率为原来的0.1倍，最小值为0.0001
Losses = nn.MSELoss()
#"""这一步是打印你的TCN模型，14为特征维度，轴承数据看你的特征列数"""
# summary(model,(sequence_length,input_channels))

###这个patience是连续150次，loss还是没有下降就停止运行
patience = 100
###这个是用于更新初始的val_loss的阈值
T_val_loss = 9999
counter = 0
train_loss = []
train_flood = []
val_loss = []
all_epochs = []
lr_step = []
###这个是模型权重的保存
save_dir = "TorchModel_TCN_MA/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    flood_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(data_train_loader):
        total = 0
        data, target = data.to(device), target.to(device)
        #data = data.view(-1, input_channels, seq_length)  # permute(0,2,1)
        # 上述一步是Torch中卷积操作独有的，如果是LSTM，transformer可以注释掉这行代码

        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze(1)
        loss = Losses(output, target)
        loss.backward()
        #         #这里添加梯度回退技巧，有一篇论文用到，你们可以试试
        #         b = torch.Tensor([145]).to(device)
        #         flood = (loss-b).abs()+b
        #         flood.backward()
        optimizer.step()

        epoch_loss += loss.item()
        #         flood_loss +=loss.item()
        total = batch_idx + 1
    epoch_loss = epoch_loss / total
    #     flood_loss = flood_loss/total
    ##############################还是要注意，不能把你要测试的数据放在这里进行验证####################################
    """不能使用是data_test_loader的数据作为  验证集，否则是算作弊
         需要做的办法：从训练集中随机，打乱抽取0.1-0.3的样本作为验证集 data_val_loader
         这个在TensorFlow中始很容易实现，但是Torch中我这一块没有研究，我自己代码是以TF2.X框架为主
         这个你们需要自己去查查，怎么制作新的validdata
    """
    min_val_loss, val_rmse = Epoch_evaluate(model, data_test_loader)  ###

    # 学习率衰减策略
    epoch_scheduler.step(min_val_loss)
    lr_current = optimizer.param_groups[0]['lr']
    lr_step.append(lr_current)

    print('epoch [{}/{}], loss:{:.4f}, val_loss:{}, val_RMSE:{}, lr:{}'.format(epoch, epochs,
                                                                               epoch_loss, min_val_loss, val_rmse,
                                                                               lr_current))  # {:.4f}
    ##这里是保存最后一个epoch的权重
    torch.save(model.state_dict(), os.path.join(save_dir, "model_on_last_epoch.pth"))

    if epoch % 1 == 0:
        train_loss.append(epoch_loss)
        val_loss.append(min_val_loss)
        all_epochs.append(epoch)
    """这里是设置训练的early stop patience """
    # 对测试集上的test_loss进行监视，并设置patience，保存在测试集上表现最好的model
    if T_val_loss >= min_val_loss:
        T_val_loss = min_val_loss
        # print("Saving...")
        best_epoch = epoch  # 找到最佳的epoch编号
        torch.save(model.state_dict(), os.path.join(save_dir, "model_onTestBest.pth"))
        counter = 0
    else:
        counter += 1
    if counter == patience:
        break

"""训练loss查看"""
plt.grid(linestyle="--")
plt.plot(all_epochs,train_loss)
plt.plot(all_epochs,val_loss)
plt.title('LN batch_size=128')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig('TorchModel/loss.png' ,bbox_inches='tight', dpi=300)  # 600
plt.show()

"""评估"""
model.load_state_dict(torch.load(os.path.join(save_dir, "model_onTestBest.pth")))
train_metrics=[]
test_metrics =[]
train_metrics = Epoch_evaluate(model,data_train_loader)
test_metrics = Epoch_evaluate(model,data_test_loader)
print('epoch [{}/{}], loss:{:.4f},     RMSE:{},'.format(best_epoch,epochs,train_metrics[0],train_metrics[1]))
print('epoch [{}/{}], val_loss:{:.4f}, val_RMSE:{}'.format(best_epoch,epochs,test_metrics[0],test_metrics[1]))

##保存评价指标：对于轴承
df = pd.DataFrame({'MSE':[train_metrics[0]] ,
                   'RMSE': [train_metrics[1]],})

df.transpose().to_csv("TorchModel/评估得分.txt",
                      mode='w',header=['train'],sep=' ',index=["MSE","RMSE"])
df = pd.DataFrame({'MSE':[test_metrics[0]] ,
                   'RMSE': [test_metrics[1]],})

df.transpose().to_csv("TorchModel/评估得分.txt",
                      mode='a',header=['test'],sep=' ',index=["MSE","RMSE"])

#RUL预测与可视化
##训练集的预测
y_train_preds,y_train_true = Last_evaluate2(model,data_train_loader)

"""保存预测的RUL值与实际的RUL值"""

y_train_All=pd.DataFrame({'Pre':y_train_preds,
                   'Actual':y_train_true})
y_train_All.to_csv("TorchModel/y_train_Pre.csv",
                      mode='w',header=["Pre","Actual"],sep=',',index=0)

##测试集的预测
y_test_preds,y_test_true = Last_evaluate2(model,data_test_loader)

"""保存预测的RUL值与实际的RUL值"""

y_All=pd.DataFrame({'Pre':y_test_preds,
                   'Actual':y_test_true})
y_All.to_csv("TorchModel/y_test_Pre.csv",
                      mode='w',header=["Pre","Actual"],sep=',',index=0)

# 可视化
"""这里轴承的可视化，不包括ma、eam、SG滤波三种平滑方法，有兴趣自己添加或者再找我"""
"""这里以XJTU-SY_Bearing_Datasets\37.5Hz11kN的数据集为例
训练集:   Bearing2-1：长度491
              Bearing2-2：长度161
              Bearing2-3：长度533
测试集:   Bearing2-4：长度42
              Bearing2-5：长度339
"""
from sklearn.metrics import mean_squared_error


def calculateScore(y_pred, y_test):
    y_pred = list(map(lambda x: x[0], y_pred))
    return mean_squared_error(y_pred, y_test)


def plotUnitLines_y_test(y_test, y_pred, unit, Bpath='', figsize=[8, 6], plotname='', dirs=None):
    mse = mean_squared_error(y_test, y_pred)

    x = range(len(y_pred))

    data_list = []

    fig = plt.figure(figsize=figsize)

    plt.plot(x, y_pred, color="red", label='原始预测', linewidth=1)
    plt.plot(x, y_test, color='blue', label='实际值', linewidth=1)
    plt.xlim(0, max(x))
    plt.ylim(0, )
    plt.grid(linestyle="--")
    plt.title('Bearing1_' + str(unit) + ', MSE: ' + str('%.5f' % mse),
              fontsize=12)
    plt.ylabel('RUL(%)', fontsize=12)
    plt.xlabel('Running Time (*10s)', fontsize=12)
    plt.legend(fontsize=12)
    fig.tight_layout()

    if len(plotname) > 0:
        save_dir = dirs + Bpath
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + plotname + ".png", bbox_inches='tight', dpi=300, format='png')  # 600
        np.savetxt(save_dir + plotname + "RULPre.csv", y_pred)
        np.savetxt(save_dir + plotname + "Act_RUL.csv", y_test)
        np.savetxt(save_dir + plotname + "x_axis.csv", x)
    plt.show()

###训练集可视化
plotUnitLines_y_test(y_train_preds[:2803], y_train[:2803], unit=1,
                     Bpath="best_epoch_result/B1_1/", figsize=[8, 6], plotname='B1_1_', dirs=save_dir)
plotUnitLines_y_test(y_train_preds[2803:2803+871], y_train[2803:2803+871], unit=2,
                     Bpath="best_epoch_result/B1_2/", figsize=[8, 6], plotname='B1_2_', dirs=save_dir)

###测试集可视化
plotUnitLines_y_test(y_test_preds[:1802], y_test[:1802], unit=1,
                     Bpath="best_epoch_result/B1_3/", figsize=[8, 6], plotname='B1_4_', dirs=save_dir)

