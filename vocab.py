# Defined in Section 4.6.1

from collections import defaultdict, Counter

class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list() #使用列表存储所有的标记，从而跟据索引值获取相应的标记
        self.token_to_idx = dict() #使用字典实现标记到索引值的映射

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        # 创建词表，输入的text包含若干句子，每个句子由若干标记构成
        token_freqs = defaultdict(int) #存储标记及其出现次数的映射词典
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        # 无重复的标记，其中预留了未登录词(Unknown word)标记(<unk>)以及若干用户自定义的预留标记
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        # 返回词表的大小
        return len(self.idx_to_token) 

    def __getitem__(self, token):
        # 查找输入标记对应的索引值，如果该标记不存在，则返回标记<unk>的索引值（0）
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标记对应的索引值
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in indices]


def save_vocab(vocab, path):
    with open(path, 'w') as writer:
        writer.write("\n".join(vocab.idx_to_token))


def read_vocab(path):
    with open(path, 'r') as f:
        tokens = f.read().split('\n')
    return Vocab(tokens)

