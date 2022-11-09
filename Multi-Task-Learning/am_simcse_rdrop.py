#! -*- coding: utf-8 -*-
import sys
import pandas as pd
from tqdm import tqdm
import re
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.constraints import unit_norm
from keras.losses import kullback_leibler_divergence as kld
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import sequence_padding
import jieba
import string
from zhon.hanzi import punctuation

# sets random seed
seed = 123
random.seed(seed)
np.random.seed(seed)
# set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set GPU memory
# 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需求增长
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
K.set_session(sess)

"""
python SimCSE/eval2.py LongforBERT cls LXH 0.3
nohup python SimCSE/eval2.py LongforBERT cls LXH 0.3 > logs/simcse.log 2>&1 &
"""

LR = 2e-5  # 1e-3
batch_size = 32  # 尽可能大
epochs = 25
t = 0.05  # 温度系数τ
scale, margin = 30, 0.35  # amsoftmax参数 0.15 0.25 0.35

# 基本参数
if len(sys.argv[1:]) == 4:
    model_type, pooling, task_name, dropout_rate = sys.argv[1:]
else:
    model_type, pooling, task_name, dropout_rate = "BERT", "cls", "LXH", 0.3
assert model_type in [
    "BERT",
    "RoBERTa",
    "NEZHA",
    "WoBERT",
    "RoFormer",
    "BERT-large",
    "RoBERTa-large",
    "NEZHA-large",
    "SimBERT",
    "SimBERT-tiny",
    "SimBERT-small",
    "LongforBERT",
]
assert pooling in ["first-last-avg", "last-avg", "cls", "pooler"]
assert task_name in ["ATEC", "BQ", "LCQMC", "PAWSX", "STS-B", "LXH"]
dropout_rate = float(dropout_rate)

# bert配置
model_name = {
    "BERT": "chinese_L-12_H-768_A-12",
    "RoBERTa": "chinese_roberta_wwm_ext_L-12_H-768_A-12",
    "WoBERT": "chinese_wobert_plus_L-12_H-768_A-12",
    "NEZHA": "nezha_base_wwm",
    "RoFormer": "chinese_roformer_L-12_H-768_A-12",
    "BERT-large": "uer/mixed_corpus_bert_large_model",
    "RoBERTa-large": "chinese_roberta_wwm_large_ext_L-24_H-1024_A-16",
    "NEZHA-large": "nezha_large_wwm",
    "SimBERT": "chinese_simbert_L-12_H-768_A-12",
    "SimBERT-tiny": "chinese_simbert_L-4_H-312_A-12",
    "SimBERT-small": "chinese_simbert_L-6_H-384_A-12",
    "LongforBERT": "longforBERT_v4.1",
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
if model_type in ["WoBERT", "RoFormer"]:
    tokenizer = get_tokenizer(
        dict_path, pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
else:
    tokenizer = get_tokenizer(dict_path)

# 建立 encoder 模型
if model_type == "RoFormer":
    pretrained_encoder = get_encoder(
        config_path,
        checkpoint_path,
        model="roformer",
        pooling=pooling,
        dropout_rate=dropout_rate,
    )
elif "NEZHA" in model_type:
    pretrained_encoder = get_encoder(
        config_path,
        checkpoint_path,
        model="nezha",
        pooling=pooling,
        dropout_rate=dropout_rate,
    )
else:
    pretrained_encoder = get_encoder(
        config_path, checkpoint_path, pooling=pooling, dropout_rate=dropout_rate
    )


def custom_ce_loss(from_logits):
    # 采用了闭包的方式，将参数传给 sparse_amsoftmax_loss，再调用 inner
    def inner(y_true, y_pred):
        return K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=from_logits
        )
    return inner


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
    # 温度系数τ=0.05
    similarities = similarities / t
    # from_logits=True的交叉熵自带softmax激活函数
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


# From: https://github.com/bojone/r-drop
def rdrop_loss(y_true, y_pred, alpha=4):
    """loss从300多开始，需要epoch=50让其下降
    """
    loss = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return K.mean(loss) / 4 * alpha


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    # loss1 = K.mean(sparse_amsoftmax_loss(y_true, y_pred, scale, margin))
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


# 模型构建
x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
target = Input(shape=(None,), dtype="int32")
emb = pretrained_encoder([x1_in, x2_in])   # pooling='cls'
emb_norm = Lambda(lambda v: K.l2_normalize(v, 1))(emb)  # 特征归一化（l2正则,专供 amsoftmax 使用）√
# emb_norm = Dropout(DROPOUT_RATE, name="dp1")(emb_norm)   #防止过拟合
output = Dense(
    cls_num,
    # activation='softmax',
    use_bias=False,  # no bias √
    kernel_constraint=unit_norm(),  # 权重归一化（单位范数（unit_form），限制权值大小为 1.0）√
    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
)(emb_norm)
r_output = Dense(
    units=cls_num,
    activation='softmax',
    kernel_constraint=unit_norm(),
    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
)(emb)

encoder = Model([x1_in, x2_in], emb)  # 最终的目的是要得到一个编码器


# 自定义的loss
# 重点：把自定义的loss添加进层使其生效，同时加入metric方便在KERAS的进度条上实时追踪
am_loss = sparse_amsoftmax_loss(target, output, scale, margin)
sim_loss = simcse_loss(target, emb)
rdrop_loss = rdrop_loss(target, r_output)
# 配合R-Drop的交叉熵损失
ce_rdrop_loss = crossentropy_with_rdrop(target, r_output)
# 配合R-Drop的amsoftmax损失
am_rdrop_loss = K.mean(am_loss) + rdrop_loss
# 配合SimCSE的amsoftmax损失
am_simcse_loss = K.mean(am_loss) + sim_loss
# All Three Loss 加权和
am_simcse_rdrop_loss = K.mean(am_loss) + sim_loss + rdrop_loss


# 自定义 metrics
def sparse_categorical_accuracy(y_true, y_pred):
    # return tf.metrics.SparseCategoricalAccuracy(y_true, y_pred[0])
    return K.metrics.sparse_categorical_accuracy(y_true, y_pred[0])


def train_rdrop():
    # 数据生成器
    train_generator = data_generator(train_token_ids, batch_size)
    # model build
    train_model = Model([x1_in, x2_in, target], [output, r_output])  # 用分类问题做训练

    # 联合训练 amsoftmax+RDrop 
    train_model.add_loss(am_rdrop_loss)
    train_model.add_metric(K.mean(am_loss), name="am_loss")
    train_model.add_metric(rdrop_loss, name="rdrop_loss")
    train_model.compile(
        optimizer=Adam(lr=LR),
        metrics=[sparse_categorical_accuracy],
        )
    custom_callback = CustomCallback(
        # valid_data=valid_data,  # (input, [kid])
        # test_data=test_data,  # (primary, kid)
        batch_size=batch_size,
        encoder=encoder,
    )
    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[custom_callback],
        # shuffle=True,
    )


def train_cse():
    # 数据生成器
    train_generator = data_generator(train_token_ids, batch_size)
    # model build
    train_model = Model([x1_in, x2_in, target], [output, emb]) 
    train_cl_model = Model([x1_in, x2_in, target], emb) 

    # 联合训练 amsoftmax+simcse
    train_model.add_loss(am_simcse_loss)
    train_model.add_metric(K.mean(am_loss), name="am_loss")
    train_model.add_metric(sim_loss, name="sim_loss")
    train_model.compile(
        optimizer=Adam(lr=LR),
        metrics=[sparse_categorical_accuracy],
        )
    custom_callback = CustomCallback(
        # valid_data=valid_data,  # (input, [kid])
        # test_data=test_data,  # (primary, kid)
        batch_size=batch_size,
        encoder=encoder,
    )
    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[custom_callback],
        # shuffle=True,
    )

    # # 数据生成器
    # train_generator = data_generator(train_token_ids, batch_size)

    # 单独 CL，缓解 bert 语义坍塌
    train_cl_model.add_loss(sim_loss)
    train_cl_model.add_metric(sim_loss, name="sim_loss")
    train_cl_model.compile(optimizer=Adam(lr=LR))
    custom_callback = CustomCallback(
        # valid_data=valid_data,  # (input, [kid])
        # test_data=test_data,  # (primary, kid)
        batch_size=batch_size,
        encoder=encoder,
    )
    train_cl_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[custom_callback],
        # shuffle=True,
    )

    # # 保存权重
    # encoder.save_weights("model/chinese_bert_simcse.h5")
    # # 加载权重测试
    # encoder.load_weights("model/chinese_bert_simcse.h5")


if __name__ == "__main__":
    train_cse()
