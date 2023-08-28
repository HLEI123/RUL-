import torch
from torch import nn
import torch.nn.functional as F
import math
class PositionalEncoding(nn.Module):
    # d_model数据特征维数，max_len最大序列长度
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


'''
将输入数据进行特征提取，添加位置信息传入Transformer

embedding_encoder:词嵌入模型，此处为autoencoder编码器
embedding_size:autoencoder输出特征维度
feature_size:Transformer输入特征维度(词嵌入输出维度)
'''


class Embedding(nn.Module):
    def __init__(self, embedding_encoder, embedding_size=32, feature_size=128, dropout=0.05):
        super(Embedding, self).__init__()
        self.embedding_encoder = embedding_encoder
        self.W_P = nn.Linear(embedding_size, feature_size)
        self.position = PositionalEncoding(feature_size)
        self.dropout = nn.Dropout()

    def forward(self, x):
        if self.embedding_encoder:
            x = self.embedding_encoder(x)  # 进行特征提取
        x = self.W_P(x)
        x = self.position(x)  # 位置编码
        x = self.dropout(x)

        return x


class Transforemer_Encoder(nn.Module):

    """
    这个就是transformer中的self attention的使用，这里是配合transformer中的Encoder一起使用
    embedding_size #原始特征维度
    feature_size  # 输入Transformer特征维度（词嵌入层输出维度）
    nhead  #  multi-head attention 的头数
    num_layers  # encoder layers 的层数
    dropout
    """
    def __init__(self, embedding_encoder, embedding_size, feature_size, nhead, num_layers, dropout):
        super(Transforemer_Encoder, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.embedding = Embedding(embedding_encoder=embedding_encoder, embedding_size=embedding_size,
                                   feature_size=feature_size)
        # 定义每一个解码器块的参数
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        # 定义整个transformer解码器的层数
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
        src = self.embedding(src)
        output = self.transformer_encoder(src, None)  # , self.src_mask)
        output = self.decoder(output)
        output = self.deco_output(output[:, -1, :])
        return F.sigmoid(output)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
