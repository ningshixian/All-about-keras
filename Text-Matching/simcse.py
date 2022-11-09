#! -*- coding: utf-8 -*-
# SimCSE 中文测试

from utils import *
import sys
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
jieba.initialize()

"""
python SimCSE/eval.py BERT cls ATEC 0.3
"""

# 基本参数
model_type, pooling, task_name, dropout_rate = sys.argv[1:]
assert model_type in [
    'BERT', 'RoBERTa', 'NEZHA', 'WoBERT', 'RoFormer', 'BERT-large',
    'RoBERTa-large', 'NEZHA-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small'
]
assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
assert task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']
dropout_rate = float(dropout_rate)

batch_size = 8  # 32
if task_name == 'PAWSX':
    maxlen = 128
else:
    maxlen = 64

# 加载数据集
data_path = 'SimCSE/senteval_cn/'

datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
    for f in ['train', 'valid', 'test']
}

# bert配置
model_name = {
    'BERT': 'chinese_L-12_H-768_A-12',
    'RoBERTa': 'chinese_roberta_wwm_ext_L-12_H-768_A-12',
    'WoBERT': 'chinese_wobert_plus_L-12_H-768_A-12',
    'NEZHA': 'nezha_base_wwm',
    'RoFormer': 'chinese_roformer_L-12_H-768_A-12',
    'BERT-large': 'uer/mixed_corpus_bert_large_model',
    'RoBERTa-large': 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16',
    'NEZHA-large': 'nezha_large_wwm',
    'SimBERT': 'chinese_simbert_L-12_H-768_A-12',
    'SimBERT-tiny': 'chinese_simbert_L-4_H-312_A-12',
    'SimBERT-small': 'chinese_simbert_L-6_H-384_A-12'
}[model_type]

config_path = "../corpus/%s/bert_config.json" % model_name
if model_type == "NEZHA":
    checkpoint_path = "../corpus/%s/model.ckpt-691689" % model_name
elif model_type == "NEZHA-large":
    checkpoint_path = "../corpus/%s/model.ckpt-346400" % model_name
else:
    checkpoint_path = "../corpus/%s/bert_model.ckpt" % model_name
dict_path = "../corpus/%s/vocab.txt" % model_name

# 建立分词器
if model_type in ['WoBERT', 'RoFormer']:
    tokenizer = get_tokenizer(
        dict_path, pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
else:
    tokenizer = get_tokenizer(dict_path)

# 建立模型
if model_type == 'RoFormer':
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='roformer',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
elif 'NEZHA' in model_type:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='nezha',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
else:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        pooling=pooling,
        dropout_rate=dropout_rate
    )

# 语料id化
all_names, all_weights, all_token_ids, all_labels = [], [], [], []
train_token_ids = []
for name, data in datasets.items():
    a_token_ids, b_token_ids, labels = convert_to_ids(data, tokenizer, maxlen)
    all_names.append(name)
    all_weights.append(len(data))
    all_token_ids.append((a_token_ids, b_token_ids))
    all_labels.append(labels)
    train_token_ids.extend(a_token_ids)
    train_token_ids.extend(b_token_ids)

if task_name != 'PAWSX':
    np.random.shuffle(train_token_ids)
    train_token_ids = train_token_ids[:10000]


class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            # 每个batch内，每一句话都重复一次
            for i in range(2):
                batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []


t = 0.05  # 温度系数τ
def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    y_true只是凑数的，并不起作用。因为真正的y_true是通过batch内数据计算得出的。
    y_pred就是batch内的每句话的embedding，通过bert编码得来
    """
    # 构造标签
    # idxs = [0,1,2,3,4,5]
    idxs = K.arange(0, K.shape(y_pred)[0])
    # 给idxs添加一个维度，idxs_1 = [[0,1,2,3,4,5]]
    idxs_1 = idxs[None, :]
    # 获取每句话的同义句id，即
    # 如果一个句子id为奇数，那么和它同义的句子的id就是它的上一句，如果一个句子id为偶数，那么和它同义的句子的id就是它的下一句
    # idxs_2 = [ [1], [0], [3], [2], [5], [4] ]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    # 生成计算loss时可用的标签
    # y_true = [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    # 首先对句向量各个维度做了一个L2正则，使其变得各项同性，避免下面计算相似度时，某一个维度影响力过大。
    y_pred = K.l2_normalize(y_pred, axis=1)
    # 其次，计算batch内每句话和其他句子的内积相似度(其实就是余弦相似度)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    # 然后，将矩阵的对角线部分变为0，代表每句话和自身的相似性并不参与运算
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    # 最后，将所有相似度乘以20，这个目的是想计算softmax概率时，更加有区分度
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


# SimCSE训练
encoder.summary()
encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
train_generator = data_generator(train_token_ids, batch_size)
encoder.fit(
    train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=1
)

# 语料向量化
all_vecs = []
for a_token_ids, b_token_ids in all_token_ids:
    a_vecs = encoder.predict([a_token_ids,
                              np.zeros_like(a_token_ids)],
                             verbose=True)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             verbose=True)
    all_vecs.append((a_vecs, b_vecs))

# 标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = l2_normalize(a_vecs)
    b_vecs = l2_normalize(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))