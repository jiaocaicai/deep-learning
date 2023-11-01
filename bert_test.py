import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from sklearn.metrics import accuracy_score, f1_score
from bert_utils import create_data_loader


# 设备
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 加载模型和权重
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=119)
model.load_state_dict(torch.load('models/bert_best_model6.pth'))
model.to(device)
model.eval()  # 如果您正在进行评估

# 参数
MAX_LEN = 512
def val(path):
    # 加载验证集上的数据
    with open(path, "r") as file:
        data = json.load(file)

    data_loader = create_data_loader(data, tokenizer, MAX_LEN, 1)
    all_predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predictions = torch.max(outputs[0], dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 4. 评估性能
    print("Accuracy:", accuracy_score(true_labels, all_predictions))
    print("f1_weighted:", f1_score(true_labels, all_predictions, average='weighted'))

print("训练集：")
val("data/trains.json")
print("验证集：")
val("data/validates.json")
print("测试集：")
val("data/tests.json")
print("训练集+验证集:")
val("trains.json")

"""
bert4:按照损失提前停止，max_len=512
训练集：
Accuracy: 0.7150490866161302
f1_weighted: 0.7087758632781478
验证集：
Accuracy: 0.5529953917050692
f1_weighted: 0.5567737537229276
测试集：
Accuracy: 0.5431578947368421
f1_weighted: 0.5454964663910816
训练集+验证集:
Accuracy: 0.6834
f1_weighted: 0.6791883166574129
"""

"""
bert5:按照f1提前停止，max_len=512
训练集：
Accuracy: 0.9840934509755188
f1_weighted: 0.9841223575565411
验证集：
Accuracy: 0.6067588325652842
f1_weighted: 0.6028020066630733
测试集：
Accuracy: 0.5957894736842105
f1_weighted: 0.5936163304570569
训练集+验证集:
Accuracy: 0.9104
f1_weighted: 0.9106818282092327
"""

"""
bert5:按照f1提前停止，max_len=450
训练集：
Accuracy: 0.9802410836336523
f1_weighted: 0.9801994976751754
验证集：
Accuracy: 0.6062467997951869
f1_weighted: 0.604953870382885
测试集：
Accuracy: 0.5894736842105263
f1_weighted: 0.591492877589408
训练集+验证集:
Accuracy: 0.9072
f1_weighted: 0.9075820452199123
"""