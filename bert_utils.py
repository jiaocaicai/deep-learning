import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import json


# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 这告诉tokenizer在序列的开始和结束位置分别添加特殊的[CLS]和[SEP]标记
            max_length=self.max_len,
            return_token_type_ids=False,  # 对于某些模型（如BERT）处理句子对时，token type IDs是有用的，但在处理单个句子时则不需要。
            truncation=True,  # 句子太长就截断
            # pad_to_max_length=True, #弃用
            padding = 'max_length',
            return_attention_mask=True,
            return_tensors='pt',  # 这告诉tokenizer返回PyTorch张量（tensors）
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 创建数据加载器
def create_data_loader(data, tokenizer, max_len, batch_size):
    texts = [entry["sentence"] for entry in data]
    labels = [int(entry["label"]) for entry in data]

    ds = TextDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, shuffle=True)
