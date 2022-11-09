import os
import re
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD, Nadam
from keras.regularizers import l2
from keras import backend as K

import tensorflow as tf
from bert4keras.models import build_transformer_model as build_bert_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_weight_decay,extend_with_exponential_moving_average
from bert4keras.optimizers import *
from bert4keras.backend import keras, set_gelu
# set_gelu('tanh')  # 切换gelu版本
from margin_softmax import sparse_amsoftmax_loss


"""sets random seed"""
seed = 123
random.seed(seed)
np.random.seed(seed)


"""
参考
https://blog.csdn.net/nima1994/article/details/83862502
https://www.pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/#pyis-cta-modal
https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
https://aistudio.baidu.com/aistudio/projectdetail/2051331?channelType=0&channel=0
https://kexue.fm/archives/7094
"""

# from util import read_tsv_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set GPU memory
# 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需求增长
sess = tf.Session(config=config)
K.set_session(sess)

# specify the batch size and number of epochs
LR = 2e-5  # [3e-4, 5e-5, 2e-5] 默认学习率是0.001
SGD_LR = 0.001
warmup_proportion = 0.1  # 学习率预热比例
weight_decay = 0.01  # 权重衰减系数，类似模型正则项策略，避免模型过拟合
DROPOUT_RATE = 0.3  # 0.1
BATCH_SIZE = 64
EPOCHS = 15
maxlen = 64  # 最大不能超过512, 若出现显存不足，请适当调低这一参数
EB_SIZE = 128
optimizer = "sgd"   # 唯独SGD有效，不知道为啥？ "sgd" "adamwlr"
kid2label, label2kid = {}, {}  # kid转换成递增的id格式

# shared layers of base network
root = r"../corpus/chinese_L-12_H-768_A-12"
tokenizer = Tokenizer(os.path.join(root, "vocab.txt"), do_lower_case=True)  # 建立分词器
bert_model = build_bert_model(
    os.path.join(root, "bert_config.json"),
    os.path.join(root, "bert_model.ckpt"),
    model="bert",
)  # bert模型不可被封装！！
drop_layer = Dropout(DROPOUT_RATE)
bn_layer = BatchNormalization()
lambda_layer = Lambda(lambda x: x[:, 0])
dense_layer = Dense(EB_SIZE, name="dense_output",kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))

# # Freeze the BERT model to reuse the pretrained features without modifying them.
# for l in bert_model.layers:
#     l.trainable = False


def consine_distance(vectors):
    (featsA, featsB) = vectors
    dis = K.sum(featsA * featsB,axis=1,keepdims=True)/(K.sum(featsA**2,axis=1,keepdims=True) * K.sum(featsB**2,axis=1,keepdims=True))
    return dis


def cross_distance(vectors):
    """借鉴sentence-bert的分类做法"""
    (featsA, featsB) = vectors
    sub = keras.layers.Lambda(lambda x: x[0] - x[1])([featsA, featsB])
    mul = keras.layers.Lambda(lambda x: x[0] * x[1])([featsA, featsB])
    distance = keras.layers.concatenate([featsA, featsB, sub, mul])
    return distance


# 两个输出向量对位相减，然后计算二次范数?
def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss


"""
https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/#pyis-cta-modal
"""
def make_pairs(images, labels):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []
	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
	# loop over all images
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		label = labels[idxA]
		# print(idxA, idx, label)
		# randomly pick an image that belongs to the *same* class
		# label
		idxB = np.random.choice(idx[label])
		if idxA == idxB:
			idxB = np.random.choice(idx[label])
		posImage = images[idxB]
		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])
		# grab the indices for each of the class labels *not* equal to
		# the current label and randomly pick an image corresponding
		# to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]
		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])
	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))


def seq_padding(ML):
    """将序列padding到同一长度, value=0, mode='post'
    """
    def func(X, padding=0):
        return np.array(
            [
                np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
                for x in X
            ]
        )
    return func


def compute_pair_input_arrays(input_arrays, maxlen, tokenizer):
    inp1_1, inp1_2, inp2_1, inp2_2 = [], [], [], []
    for instance in input_arrays:
        a, b = instance[0], instance[1]
        x1, x2 = tokenizer.encode(a, maxlen=maxlen)
        inp1_1.append(x1)
        inp1_2.append(x2)
        y1, y2 = tokenizer.encode(b, maxlen=maxlen)
        inp2_1.append(y1)
        inp2_2.append(y2)

    L = [len(x) for x in inp1_1+inp2_1]
    ML = max(L) if L else 0
    
    pad_func = seq_padding(ML)
    res = [inp1_1, inp1_2, inp2_1, inp2_2]
    res = list(map(pad_func, res))
    return res


# def clean(x):
#     """预处理：去除文本的噪声信息
#     """
#     x = re.sub('"', "", x)
#     x = re.sub("\s", "", x)  # \s匹配任何空白字符，包括空格、制表符、换页符等
#     x = re.sub(",", "，", x)
#     return x

def clean_sim(x):
    x = re.sub(r"(\t\n|\n)", "", x)
    x = x.strip().strip("###").replace("######", "###")
    return x.split("###")


def create_test_pairs(test):
    """Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    pairs = []
    recall_ids = []
    answer_ids = []
    for index, row in test.iterrows():
        query = row["user_input"]
        answer_id = clean_sim(row["answer_id"])
        recall = clean_sim(row["recall"])
        recall_id = clean_sim(row["recall_id"])
        try:
            assert len(recall) == len(recall_id) == 10
        except:
            print(index)
            print(query)
            print(recall)
            print(recall_id)
            continue
        for item, kid in zip(recall, recall_id):
            pairs.append([query, item])
            recall_ids.append(kid)
            answer_ids.append(answer_id)
    return (np.array(pairs), np.array(recall_ids), np.array(answer_ids))


# def compute_input_arrays_pairsent(input_arrays, maxlen, tokenizer=None):
    
#     def seq_padding(X, padding=0):
#         L = [len(x) for x in X]
#         ML = max(L) if L else 0
#         return np.array(
#             [
#                 np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
#                 for x in X
#             ]
#         )

#     x1_, x2_ = [], []
#     for instance in tqdm(input_arrays):
#         token_ids, segment_ids = tokenizer.encode(
#             instance[0], instance[1], maxlen=maxlen
#         )
#         x1_.append(token_ids)
#         x2_.append(segment_ids)
#     x1_ = seq_padding(x1_)
#     x2_ = seq_padding(x2_)
#     return [x1_, x2_]


# def create_base_network(embeddingDim=128):
#     """Base network to be shared (eq. to feature extraction).
#     """
#     # Input for anchor, positive and negative images
#     x1_in = Input(shape=(None,))
#     x2_in = Input(shape=(None,))

#     # Output for anchor, positive and negative embedding vectors
#     # The bert_model instance is shared (Siamese network)
#     x = bert_model([x1_in, x2_in])
#     cls_embedding = Lambda(lambda x: x[:, 0])(x)  # first_token
    
#     # # passed into an FC layer that has 128 nodes.
#     # cls_embedding = Dense(embeddingDim)(cls_embedding)
#     return Model([x1_in, x2_in], cls_embedding)


def build_siamese_model():
	# configure the siamese network
    print("[INFO] building siamese network...")
    x1_in = Input(shape=(None,), name="anchor_input")
    x2_in = Input(shape=(None,))
    z1_in = Input(shape=(None,), name="sample_input")
    z2_in = Input(shape=(None,))
    # Base network to be shared (eq. to feature extraction).
    featsA = bert_model([x1_in, x2_in])
    featsA = lambda_layer(featsA)  # first_token
    # featsA = drop_layer(featsA)
    # featsA = Lambda(lambda x: K.l2_normalize(x, 1))(featsA)
    featsA = dense_layer(featsA)    # 降维，减少距离度量的计算量
    featsA = bn_layer(featsA)
    featsB = bert_model([z1_in, z2_in])
    featsB = lambda_layer(featsB)  # first_token
    # featsB = drop_layer(featsB)
    # featsB = Lambda(lambda x: K.l2_normalize(x, 1))(featsB)
    featsB = dense_layer(featsB)    # 降维，减少距离度量的计算量
    featsB = bn_layer(featsB)

    # 语义匹配任务: 相似、不相似 2 分类任务
    distance = Lambda(euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)
    # distance = Lambda(cross_distance)([featsA, featsB])
    # outputs = Dense(2, kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(distance)    # , activation="softmax"

    model = Model(inputs=[x1_in, x2_in, z1_in, z2_in], outputs=outputs)
    model.summary()
    return model


def compute_accuracy(y_true, y_pred):  # numpy上的操作
    """Compute classification accuracy with a fixed threshold on distances.
    """
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):  # Tensor上的操作
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	# plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	# plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, batch_size=16, encoder=None):
        # self.valid_inputs = valid_data  # (input, [kid])
        # self.valid_outputs = []  # valid_data[-1]
        # self.test_inputs = test_data  # (primary, kid)
        self.batch_size = batch_size
        self.encoder = encoder
        self.f1 = 0

    def evaluate(self, probs, prediction, y):
        probs = sorted(probs, key=lambda x: -x[0])
        ind = probs[0][1]  # 概率最大的候选索引
        top1_id, kid = te_ids[ind], te_y[ind]
        flag = False
        if top1_id in kid:
            prediction.append(top1_id)
            y.append(top1_id)
            flag = True
        else:
            prediction.append(top1_id)
            y.append(kid[0])
        return flag, top1_id, kid

    def on_train_begin(self, logs={}):
        self.valid_predictions = []

    def on_epoch_end(self, epoch, logs={}):
        # res_pred = self.encoder.predict([np.array(test_x[0]), np.array(test_x[1])], batch_size=BATCH_SIZE, verbose=1)
        res_pred = self.model.predict(test_x, batch_size=BATCH_SIZE, verbose=1)
        res_pred = [x[0] for x in res_pred]
        assert len(res_pred) == len(te_ids) == len(te_y)

        prev = te_pairs[0][0]
        probs, groups = [], []
        hits, total = 0, 0
        prediction, y = [], []
        with open("train_data/prediction.txt", "w", encoding="utf-8") as f:
            for i in tqdm(range(len(te_ids))):
                recall_id, answer_id, sim_prob = te_ids[i], te_y[i], res_pred[i]
                query, candidate = te_pairs[i][0], te_pairs[i][1]

                if not query == prev:
                    flag, top1_id, kid = self.evaluate(probs, prediction, y)
                    if flag:
                        hits += 1
                    total += 1
                    f.write("\n".join(groups))
                    f.write("\ntop1_id: " + str(top1_id) + "\tkid: " + str(kid))
                    f.write("\n\n")
                    probs, groups = [], []

                prev = query
                probs.append((sim_prob, i))
                groups.append("\t".join([query, candidate, recall_id, str(sim_prob)]))

            if probs:
                flag, top1_id, kid = self.evaluate(probs, prediction, y)
                if flag:
                    hits += 1
                total += 1
                f.write("\n".join(groups))
                f.write("\ntop1_id: " + str(top1_id) + "\tkid: " + str(kid))
                f.write("\n\n")

        print("测试集：{}   命中个数：{}    accuracy：{}".format(total, hits, hits / total))
        report = classification_report(prediction, y, digits=4, output_dict=True)
        # print("\nTop1 micro avg", report["accuracy"])  # 0.7378378378378379
        print("Top1 macro avg", report["macro avg"])
        print("Top1 weighted avg", report["weighted avg"])

        # f1 = f1_score(prediction, y, average="weighted")
        # if f1 > self.f1:
        #     print("epoch:{} 当前最佳f1-score！\n".format(epoch))
        #     self.f1 = f1
        #     self.model.save_weights("nlu_sort_siamese_best.h5")


# load Longfor ROBOT dataset 
# df.columns = ['knowledge_id','question','base_code','category_id']
df = pd.read_csv("train_data/train.csv",header=0,sep=',', encoding="utf-8", engine="python")
for kid in df['knowledge_id']:
    kid2label.setdefault(kid, len(kid2label))
label2kid = {v:k for k,v in kid2label.items()}
x_train, y_train = np.array(df['question']), np.array(list(map(lambda x: kid2label[x], df['knowledge_id'])))
x_test, y_test = x_train[:100], y_train[:100]

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(x_train, y_train)
(pairTest, labelTest) = make_pairs(x_test, y_test)
with open("train_data/train_pair.txt", "w", encoding="utf-8") as f:
    for a, b in zip(pairTrain, labelTrain):
        f.write(str(a) + "\t" + str(b) + "\n")

# # 加入外部[千言文本相似度的LCQMC数据】
# # 更快收敛，效果无明显提升
# pairTrain2, labelTrain2 = [], []
# with open("SimCSE/senteval_cn/LCQMC/LCQMC.train.data", encoding='utf-8') as f:
#     for l in f:
#         l = l.strip().split('\t')
#         if len(l) == 3:
#             pairTrain2.append([l[0], l[1]])
#             labelTrain2.append([int(l[2])])
# pairTrain, labelTrain = np.append(pairTrain, pairTrain2, axis=0), np.append(labelTrain, labelTrain2, axis=0)

pairTrain = compute_pair_input_arrays(pairTrain, maxlen, tokenizer=tokenizer)
# pairTest = compute_pair_input_arrays(pairTest, maxlen, tokenizer=tokenizer)
print(pairTrain[0].shape, pairTrain[1].shape)  # (142960, 64) (142960, 64)
print(pairTrain[2].shape, pairTrain[3].shape)  # (142960, 64) (142960, 64)
# print(pairTrain.shape)  # (4, 142960, 189)
# print(pairTest.shape)  # (4, 200, 63)
# print(pairTrain[0][:4])
# print(pairTrain[2][:4])

# df_test.columns = ['user_input','answer','answer_id','recall', 'recall_id']
df_test = pd.read_csv("train_data/it_test.csv",header=0,sep=',', encoding="utf-8", engine="python")  # 防止乱码
te_pairs, te_ids, te_y = create_test_pairs(df_test)
# te_pairs, te_ids, te_y = te_pairs[:100], te_ids[:100], te_y[:100]
test_x = compute_pair_input_arrays(te_pairs, maxlen, tokenizer=tokenizer)
print(len(test_x), test_x[0].shape)  # 4 (3600, 64)


# # create training+test positive and negative pairs
# num_classes = 10
# digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
# tr_pairs, tr_y = create_pairs(x_train, digit_indices)
# digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
# te_pairs, te_y = create_pairs(x_test, digit_indices)


# configure the siamese network
model = build_siamese_model()

# 优化器选择
if optimizer.lower() == 'sgd':
    opt = SGD(lr=SGD_LR, decay=1e-5, momentum=0.9, nesterov=True)
    # opt = SGD(lr=1e-3, decay=1e-3/EPOCHS, momentum=0.9, nesterov=True)
elif optimizer.lower() == 'adam':
    opt = Adam(lr=LR, clipvalue=1.)
elif optimizer.lower() == 'nadam':
    opt = Nadam(lr=LR, clipvalue=1.)
elif optimizer.lower() == 'rmsprop':
    opt = RMSprop(lr=LR, clipvalue=1.)
elif optimizer.lower() == 'adamw':
    # 变成带权重衰减的Adam
    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    opt = AdamW(LR, weight_decay_rate=weight_decay)
elif optimizer.lower() == 'adamlr':
    # 变成带分段线性学习率的Adam
    AdamLR = extend_with_piecewise_linear_lr(Adam, 'AdamLR')
    # 实现warmup，前1000步学习率从0增加到0.001
    opt = AdamLR(learning_rate=LR, lr_schedule={1000: 1.})
elif optimizer.lower() == 'adamga':
    # 变成带梯度累积的Adam
    AdamGA = extend_with_gradient_accumulation(Adam, 'AdamGA')
    opt = AdamGA(learning_rate=LR, grad_accum_steps=10)
elif optimizer.lower() == 'adamla':
    # 变成加入look ahead的Adam
    AdamLA = extend_with_lookahead(Adam, 'AdamLA')
    opt = AdamLA(learning_rate=LR, steps_per_slow_update=5, slow_step_size=0.5)
elif optimizer.lower() == 'adamlo':
    # 变成加入懒惰更新的Adam
    AdamLO = extend_with_lazy_optimization(Adam, 'AdamLO')
    opt = AdamLO(learning_rate=LR, include_in_lazy_optimization=[])
elif optimizer.lower() == 'adamwlr':
    # 组合使用
    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, 'AdamWLR')
    # 带权重衰减和warmup的优化器
    opt = AdamWLR(learning_rate=LR,
                        weight_decay_rate=weight_decay,
                        lr_schedule={1000: 1.})
elif optimizer.lower() == "adamema":
    AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
    opt = AdamEMA(learing_rate, ema_momentum=0.9999)

# compile the model
print("[INFO] compiling model...")
model.compile(
    # "binary_crossentropy", sparse_categorical_crossentropy, sparse_amsoftmax_loss, contrastive_loss
    loss=binary_crossentropy, 
    optimizer=opt,
	metrics=["accuracy", 'sparse_categorical_accuracy'])

# train the model
print("[INFO] training model...")
custom_callback = CustomCallback(batch_size=BATCH_SIZE)
history = model.fit(
	x=pairTrain, 
    y=labelTrain[:],
	# validation_data=(pairTest, labelTest[:]),
	batch_size=BATCH_SIZE, 
	epochs=EPOCHS, 
    shuffle=True,
    callbacks=[custom_callback],
)

# # serialize the model to disk
# print("[INFO] saving siamese model...")
# model.save(config.MODEL_PATH)
# plot the training history
print("[INFO] plotting training history...")
plot_training(history, "plot.png")


# # compute final accuracy on training and test sets
# y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
# tr_acc = compute_accuracy(tr_y, y_pred)
# y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# te_acc = compute_accuracy(te_y, y_pred)

# print("* Accuracy on training set: %0.2f%%" % (100 * tr_acc))
# print("* Accuracy on test set: %0.2f%%" % (100 * te_acc))
