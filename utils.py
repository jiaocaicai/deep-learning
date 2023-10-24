# Defined in Section 4.6.4

import torch
from vocab import Vocab
def seg():
    import jieba
    import json

    # 10000条训练数据
    with open('trains.json', 'r',encoding='utf-8') as f:
        trains = json.load(f)
        
    train_data = [] #[[第一个句子的分词结果],[第二个句子的分词结果],...]
    for sent in trains:
        sentence = sent['sentence'] if len(sent['sentence'])<=512 else sent['sentence'][:512]  #保证句子长度小于等于512
        fenci = jieba.cut(sentence,cut_all=False)
        res=[]
        for j in fenci:
            res.append(j)
        train_data.append(res)

        
    # 1425条测试数据
    with open('tests.json', 'r',encoding='utf-8') as f:
        tests = json.load(f)
    
    test_data = [] #[[第一个句子的分词结果],[第二个句子的分词结果],...]
    for sent in tests:
        sentence = sent['sentence'] if len(sent['sentence'])<=512 else sent['sentence'][:512]
        fenci = jieba.cut(sentence,cut_all=False)
        res=[]
        for j in fenci:
            res.append(j)
        test_data.append(res)  

    return trains, tests, train_data, test_data

def load_sentence():
    trains,tests,train_data,test_data = seg()

    vocab = Vocab.build(train_data + test_data)

    for i,sent in enumerate(trains):
        train_data[i] = (vocab.convert_tokens_to_ids(train_data[i]), int(sent['label']))
    
    for i,sent in enumerate(tests):
        test_data[i] = (vocab.convert_tokens_to_ids(test_data[i]), int(sent['label']))

    return train_data, test_data, vocab

def length_to_mask(lengths):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask


