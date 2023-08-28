from data_provider.data_loader import sequence_data_loader
import sys
sys.path.append("...")
from models.Autoencoder import AE
from models.Transformer import Transforemer_Encoder
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch import nn
plt.rcParams['font.sans-serif'] = 'Simsun'
plt.rcParams.update({'font.size': 12}) # 改变所有字体大小，改变其他性质类似

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

feature_size=27
data_path="PHM2012/时域频域特征提取/退化比例0.5/"
header=None
output_feature_size=27
sequence_length=64

data_train_loader,data_val_loader,data_test_loader=sequence_data_loader(feature_size,data_path,header,output_feature_size,sequence_length,degradation_ratio=0.8)


autoencoder = AE()
autoencoder.load_state_dict(torch.load("autoencoder_model.pth"))
autoencoder.to(device)
autoencoder.train()
###设置transformer的基本参数


# define and load model
model = Transforemer_Encoder(embedding_encoder=None,embedding_size=27,feature_size=256,
                             nhead=2, num_layers=2, dropout=0.25).to(device)


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
        return round(min_val_loss, 5), np.round(val_rmse, 5)

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
        return np.round(preds, 4), np.round(true, 5)



epochs = 1
steps = 0

# initialize Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=50,min_lr=0.001)###设置学习率下降策略，连续50次loss没有下降，学习率为原来的0.1倍，最小值为0.0001
Losses = nn.MSELoss()


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
save_dir = "model/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    flood_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(data_train_loader):
        total = 0
        data, target = data.to(device), target.to(device)
        # data = data.view(-1, input_channels, seq_length)#permute(0,2,1)
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
    min_val_loss, val_rmse = Epoch_evaluate(model, data_val_loader)  ###

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

