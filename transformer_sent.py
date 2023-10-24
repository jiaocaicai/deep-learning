# Defined in Section 4.6.8

import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import defaultdict
from vocab import Vocab
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from utils import load_sentence, length_to_mask

# tqdm是一个Pyth模块，能以进度条的方式显式迭代的进度
from tqdm.auto import tqdm
import copy

class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 对偶数位置编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 对奇数位置编码
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe是一个tensor对象，它被注册为模型的缓冲区，并且被命名为pe。这意味着在模型中可以通过self.pe来访问和使用这个缓冲区。缓冲区是一种特殊的tensor，它可以与模型的参数一起保存和加载，并且在模型的前向传播过程中不会被更新。即不对位置编码层求梯度。
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 注意：如果x的第一维比self.pe大呢？？？应该要从把原始数据截断吧
        x = x + self.pe[:x.size(0), :]  # 输入的词向量与位置编码相加
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1, max_len=512, activation: str = "relu"):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)
        # 编码层：使用TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出层
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        # 与LSTM处理情况一样，输入数据的第1维是批次，需要转换为TransformerEncoder，所需的第1维是长度，第2维是批次的形状
        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)
        attention_mask = length_to_mask(lengths) == False  # 根据批次中每个序列的长度生成Mask矩阵
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]  # 第一个隐含层（代表整个序列）
        output = self.output(hidden_states)  # 取第一个标记的输出结果作为分类层的输入
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


embedding_dim = 128
hidden_dim = 128
num_class = 119
batch_size = 32
num_epoch = 5

# 加载数据
train_data, test_data, vocab = load_sentence()
train_data, validate_data = train_test_split(train_data, test_size=0.2, random_state=42)  # 划分训练集和验证集
train_dataset = TransformerDataset(train_data)
validate_dataset = TransformerDataset(validate_data)
test_dataset = TransformerDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
validate_data_loader = DataLoader(validate_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

# 加载模型
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = Transformer(len(vocab), embedding_dim, hidden_dim, num_class, num_head=4, num_layers=2)
model.to(device)  # 将模型加载到GPU中（如果已经正确安装）

# 训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# model.train()

best_epoch = None
best_model = None
min_val_loss = np.float64('inf')
for epoch in range(1,num_epoch+1):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        log_probs = model(inputs, lengths)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += (loss.item()*batch_size)
    print(f"TrainLoss: {total_loss/len(train_data):.2f}",end='\t\t')

    validate_total_loss = 0
    for batch in validate_data_loader:
        inputs, lengths, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            validate_total_loss += loss.item()
    print(f"ValidateLoss: {validate_total_loss/len(validate_data):.2f}")

    if validate_total_loss < min_val_loss:
        min_val_loss = validate_total_loss
        best_epoch = epoch
        best_model = model

import pickle

# save the model
with open('transformer_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
# load the saved model
# with open('transformer_model.pkl', 'rb') as f:
#     model = pickle.load(f)
# predict using the loaded model


# 测试过程
acc = 0
test_total_loss = 0
y_pred = np.array([]).astype(int)  # 存储真实的target
y_true = np.array([]).astype(int)  # 存储预测出来的target
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = best_model(inputs, lengths)
        loss = nll_loss(output, targets)
        y_pred = np.append(y_pred, output.argmax(dim=1).cpu().numpy())  # 要先把数据放在cpu上，才能转成numpy
        y_true = np.append(y_true, targets.cpu().numpy())
        acc += (output.argmax(dim=1) == targets).sum().item()
        test_total_loss += loss.item()

# 输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")
print(f"f1_macro:{f1_score(y_true, y_pred, average='macro'):.2f}")
print(f"f1_weighted:{f1_score(y_true, y_pred, average='weighted'):.2f}")
print(f"f1_macro:{f1_score(y_true, y_pred, average='macro'):.2f}")
print(f"TestLoss:{test_total_loss}")