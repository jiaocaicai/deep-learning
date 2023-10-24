# Defined in Section 4.6.7

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import defaultdict
from vocab import Vocab
from utils import load_sentence
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# tqdm是一个Python模块，能以进度条的方式显式迭代的进度
from tqdm.auto import tqdm


class LstmDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    # 获得每个序列的长度
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        # pack_padded_sequence函数将变长序列打包
        # lengths.cpu()注意这里改为cpu，pytorch自身的问题(自己加的)
        x_pack = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        outputs = self.output(hn[-1])
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs


embedding_dim = 128
hidden_dim = 256
num_class = 119
batch_size = 32
num_epoch = 20 #5

# 加载数据
train_data, test_data, vocab = load_sentence()
train_dataset = LstmDataset(train_data)
test_dataset = LstmDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

# 加载模型
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device)  # 将模型加载到GPU中（如果已经正确安装）

# 训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        #inputs, lengths, targets = [x for x in batch]
        log_probs = model(inputs, lengths)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

import pickle
# save the model
with open('lstm_model.pkl', 'wb') as f:
    pickle.dump(model, f)
# load the saved model
# with open('lstm_model.pkl', 'rb') as f:
#     model = pickle.load(f)
# predict using the loaded model
#model.predict(X)

# 测试过程
acc = 0
y_pred = np.array([]).astype(int)  # 存储真实的target
y_true = np.array([]).astype(int)  # 存储预测出来的target
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets = [x.to(device) for x in batch]
    #inputs, lengths, targets = [x for x in batch]
    with torch.no_grad():
        output = model(inputs, lengths)
        y_pred = np.append(y_pred, output.argmax(dim=1).cpu().numpy())  # 要先把数据放在cpu上，才能转成numpy
        y_true = np.append(y_true, targets.cpu().numpy())
        acc += (output.argmax(dim=1) == targets).sum().item()

# 输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")
print(f"f1_macro:{f1_score(y_true, y_pred, average='macro'):.2f}")
print(f"f1_weighted:{f1_score(y_true, y_pred, average='weighted'):.2f}")
print(f"f1_macro:{f1_score(y_true, y_pred, average='macro'):.2f}")
