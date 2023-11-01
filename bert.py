import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import json
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from bert_utils import create_data_loader

MAX_LEN = 450  #450
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-5

# train_data, validate_data = train_test_split(train_data, test_size=0.2, random_state=42)  # 划分训练集和验证集

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=119)


# 加载划分好的数据
with open("data/trains.json", "r") as file:
    train_data = json.load(file)

with open("data/validates.json", "r") as file:
    validate_data = json.load(file)

train_data_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
validate_data_loader = create_data_loader(validate_data, tokenizer, MAX_LEN, BATCH_SIZE)


# 训练准备
# 设备
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 优化器
# get_linear_schedule_with_warmup：函数创建了一个学习率调度器。
# num_warmup_steps：这是学习率在训练期间将线性增加从0到指定学习率（优化器中的lr）的优化步数（批次数）。在这种情况下，它设置为0，因此没有热身阶段，学习率从一开始就是lr。
# num_training_steps：调度器将在这些步数上调整学习率
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_data_loader) * EPOCHS #一共要调整学习率的次数
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 损失函数：训练数据的类别分布不平衡时，模型可能会偏向于更多的类别，导致性能下降。所以为每个类分配一个权重，使得少数类得到更高的权重
from sklearn.utils.class_weight import compute_class_weight
labels = [int(entry["label"]) for entry in train_data]
class_weights = compute_class_weight('balanced', classes=list(set(labels)), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # 注意这里的dtype=torch.float
# 类权重结合交叉熵损失
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

# 训练
best_epoch = None
max_val_f1 = 0
PATIENCE = 3  # 提前停止的条件
no_improve = 0
for epoch in range(EPOCHS):
    for batch in tqdm(train_data_loader,desc=f"Training Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        #outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) #加labels就会用默认的损失：交叉熵损失（cross-entropy loss）
        #loss = outputs[0]  # 它将返回一个元组，其中第一个元素是交叉熵损失（cross-entropy loss）

        outputs = model(input_ids=input_ids, attention_mask=attention_mask) #不加labels就不会用默认的损失
        logits = outputs[0]
        loss = loss_function(logits ,labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print(f"TrainLoss: {loss:.4f}",end='\t')

    validate_loss = 0
    all_predictions = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        for batch in validate_data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            #outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            #validate_loss += outputs[0].item()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            validate_loss += loss_function(logits,labels).item()

            _, predictions = torch.max(outputs[0], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            validationLoss = validate_loss/len(validate_data_loader)
            accuracy = accuracy_score(true_labels, all_predictions)
            f1 = f1_score(true_labels, all_predictions, average='weighted')
        print(f"ValidationLoss: {validationLoss:.4f} \t Accuracy:{accuracy} \t f1_weighted:{f1}")

    if max_val_f1 < f1:
        max_val_f1 = f1
        best_epoch = epoch
        torch.save(model.state_dict(), 'models/bert_best_model6.pth')
        no_improve = 0
    else:
        no_improve += 1

    if no_improve == PATIENCE:
        print("Early stopping due to no improvement.")
        break

# 保存模型参数
#torch.save(model.state_dict(), 'weights_path.pth')
# 加载权重时
# model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(set([entry["label"] for entry in train_data])))
# model.load_state_dict(torch.load('weights_path.pth'))
# model.eval()  # 如果您正在进行评估