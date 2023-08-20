import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from util.add_timestep import gen_test,Scoring_2008
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt

max_abs_scaler=MaxAbsScaler()

def constructSignal(raw_signal,train_x):
    for i in range((len(raw_signal)-1280)//100):
        train_x.append(raw_signal[i*100:i*100+1280])

#读取训练集
index_columns_names=['UnitID','RUL','ScaleRUL']

features_colums=['s'+str(i) for i in range(1,2561)]
train_features_colums=['s'+str(i) for i in range(1,1281)]
features_colums1=['s'+str(i) for i in range(1,1281)]
input_file_column_names=index_columns_names+features_colums
path_dir="PHM2012/原始2560维振动数据/bearing1_1.csv"
data=pd.read_csv(path_dir,names=input_file_column_names,header=None)
data=max_abs_scaler.fit_transform(data[features_colums])
train_x=[]

for i in range(len(data)):
    print("第{}行".format(i))
    constructSignal(data[i,:].tolist(),train_x)


train_x=torch.Tensor(train_x)
# train_x=train_x.unsqueeze(1)
path_dir="PHM2012/原始2560维振动数据/bearing1_2.csv"
data=pd.read_csv(path_dir,names=input_file_column_names,header=None)

test_x=data[train_features_colums]
test_x=torch.tensor(test_x.values,dtype=torch.float32)
# test_x=test_x.unsqueeze(1)
batch_size=128
train_x=TensorDataset(train_x)
train_x=DataLoader(train_x,batch_size,shuffle=True)
test_x=TensorDataset(test_x)
test_x=DataLoader(test_x,batch_size,shuffle=True)


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # [b, 784] => [b, 20]
        self.encoder = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        # [b, 20] => [b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1280),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: [b, 1, 28, 28]
        :return:
        """
        batch_size = x.size(0)
        # flatten
        x = x.view(batch_size, 1280)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(batch_size,1280)

        return x, None


device = torch.device('cuda')
model = AE().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

print(model)
epoch_loss=[]
for epoch in range(10):
    model.train()
    train_loss=[]
    for x in train_x:
        x = x[0].to(device)
        x_hat, _ = model(x)
        loss = criterion(x_hat, x)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    print(f'epoch : {epoch},loss : {loss.item()}')
    epoch_loss.append(np.average(train_loss))

    x= iter(test_x).next()
    model.eval()
    x = x[0].to(device)
    with torch.no_grad():
        x_hat, _ = model(x)

plt.plot(range(len(epoch_loss)),epoch_loss)
torch.save(model.state_dict(), 'autoencoder_model.pth')