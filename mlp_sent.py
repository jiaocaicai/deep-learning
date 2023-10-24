# Defined in Section 4.6.4 and 4.6.5

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from vocab import Vocab
from utils import load_sentence

class BowDataset(Dataset):
    def __init__(self, data): #data为原始的数据
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def collate_fn(examples):
    # 从独立样本集合中构建各批次的输入输出
    # 其中，BowDatasete类定义了一个样本的数据结构，即输入标签和输出标签的元组
    # 因此，将输入inputs定义为一个张量的列表，其中每个张量为原始句子中标记序列对应的索引值序列ex[0]
    inputs = [torch.tensor(ex[0]) for ex in examples]
    # 输出目标targets为该批次中全部样例输出结果（0或1）构成的张量
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    offsets = [0] + [i.shape[0] for i in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  #获取一个批次中每个序列的偏移量
    inputs = torch.cat(inputs)
    return inputs, offsets, targets

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)
    def forward(self, inputs, offsets):
        embedding = self.embedding(inputs, offsets)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs

# tqdm是一个Python模块，能以进度条的方式显示迭代的进度
#from tqdm.auto import tqdm  #不知道为什么这个不显示进度条和描述信息
from tqdm import *

# 超参数设置
embedding_dim = 128
hidden_dim = 256
num_class = 2
batch_size = 32
num_epoch = 5

# 加载数据
train_data, test_data, vocab = load_sentence()
train_dataset = BowDataset(train_data) #BowDataset是Dataset的子类
test_dataset = BowDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device) # 将模型加载到CPU或GPU设备

#训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, offsets, targets = [x.to(device) for x in batch]
        log_probs = model(inputs, offsets)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

# 测试过程
acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, offsets, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, offsets)
        acc += (output.argmax(dim=1) == targets).sum().item()

# 输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")
