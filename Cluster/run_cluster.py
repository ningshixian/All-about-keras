# -*- coding:utf-8 -*-
"""
@author: ningshixian
@file: main.py
@time: 2019/7/23 15:08
"""
import os
import pickle
import numpy as np
import joblib
import string
import re
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist

# from zhon.hanzi import punctuation  # 中文标点符号
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import codecs
from tqdm import tqdm
import run_canopy as canopy


# 设置matplotlib正常显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

kmeans_path = "model/kmeans"
encoding = "utf-8"
# 聚类数据的保存路径
RUN_CLUSTER_PKL = "data/run_cluster.pkl"


# 将聚类结果写入文件
def writeClusterResult(y):
    result_dict = {}
    for i in range(len(y)):
        if y[i] not in result_dict:
            result_dict[y[i]] = []
        result_dict[y[i]].append(data[i].strip())

    # 按value长度逆序排序，写入
    result_list = sorted(result_dict.items(), key=lambda d: len(d[1]), reverse=True)
    with codecs.open("result/prediction_cluster.txt", "w", encoding=encoding) as target:
        for item in result_list:
            target.write("{}\n{}\n\n".format(item[0], "###".join(item[1])))
    print('聚类结果写入文件完成!!!')


# 对聚类结果进行可视化
def visualization(kmeans_path, X_tsne):
    cluster = joblib.load(kmeans_path)
    y = cluster.predict(X_tsne).tolist()

    # 中心点
    cents = cluster.cluster_centers_  # 质心
    # t-SNE降维转换后的输出
    # cents = getPCAData(cents,2)
    # cents = tsne.fit_transform(cents)
    # 每个样本所属的簇
    labels = cluster.labels_
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    sse = cluster.inertia_

    # 画出聚类结果，每一类用一种颜色
    colors = ["b", "g", "r", "k", "c", "m", "y", "#e24fff", "#524C90", "#845868"]
    # 标识出质心("D"表示形状为菱形，后面表示不同的颜色)
    mark = ["Dr", "Db", "Dg", "Dk", "^b", "+b", "sb", "db", "<b", "pb"]

    for i in range(best_K):
        index = np.nonzero(labels == i)[0]
        x0 = X_tsne[index, 0]
        x1 = X_tsne[index, 1]
        y_i = y[index]
        ind = i - len(colors) * (i // len(colors) + 1)  # 颜色的索引
        for j in range(len(x0)):
            plt.text(
                x0[j],
                x1[j],
                str(int(y_i[j])),
                color=colors[ind],
                fontdict={"weight": "bold", "size": 9},
            )
        # plt.plot(cents[i][0], cents[i][1], mark[i], markersize=12)
        plt.scatter(
            cents[i, 0], cents[i, 1], marker="x", color=colors[ind], linewidths=12
        )

    plt.grid(True, linestyle=":", color="r", alpha=0.6)  # 设置网格刻度
    plt.title("闲聊问题的Kmeans聚类结果")  # 显示图标题
    plt.axis([-60, 60, -60, 60])
    plt.savefig("result/kmeans_tsne.png")
    plt.show()


# 1、读取run_classifier.py的预测结果
predictions_path = "result/predictions.txt"
data = []
garbled = 0
with codecs.open(predictions_path, encoding=encoding) as f:  # , errors='ignore'
    for line in f:
        line = line.strip()
        # 去除乱码
        try:
            line.encode(encoding="gbk", errors="strict")
        except UnicodeEncodeError as ue:
            garbled += 1
            continue
        # 替换空格为逗号
        line = line.replace("   ", "，").replace("  ", "，")
        if line:  # 确保不为空
            data.append(line)
print("包含乱码的问题数: {}".format(garbled))

# with codecs.open('result/predictions_new.txt', 'w' ,encoding=encoding) as f:
#     for line in data:
#         f.write(line + '\n')
# exit()


print("#----------------------------------------#")
print("#                                        #")
print("#           2、降维&&KMeans聚类           #")
print("#                                        #")
print("#--------------------------------------#\n")

print("开启BERT服务，对句子进行向量化")
from bert_serving.client import BertClient

bc = BertClient(
    ip="10.240.4.47",
    port=5555,
    port_out=5556,
    timeout=-1,
    check_length=False,
    check_version=False,
    check_token_info=False,
)

if os.path.exists(RUN_CLUSTER_PKL):
    with codecs.open(RUN_CLUSTER_PKL, "rb") as f:
        X = pickle.load(f)
else:
    # 获取句子向量
    X = bc.encode(data)
    with codecs.open(RUN_CLUSTER_PKL, "wb") as f1:
        pickle.dump(X, f1)
print("完成!")

# # 1) t-SNE降维, 用于可视化 (费时)
# tsne = TSNE(n_components=2, init="pca", random_state=0)
# X_tsne = tsne.fit_transform(X)  # 转换后的输出

# 2) 主成分分析 数据降维度
from sklearn.decomposition import PCA
def getPCAData(data,comp):
    pcaClf = PCA(n_components=comp, whiten=True)
    pcaClf.fit(data)
    data_PCA = pcaClf.transform(data) # 用来降低维度
    return data_PCA
X_tsne = getPCAData(X,2)


# # 若已存在训练好的聚类模型
# if os.path.exists(kmeans_path):
#     print("加载已保存模型...")
#     cluster = joblib.load(kmeans_path)
#     # 直接对数据进行聚类
#     y = cluster.predict(X_tsne).tolist()
#     # 将聚类结果写入文件
#     writeClusterResult(y)
#     exit()  # 不再往下执行


# [Canopy算法]事前最佳 K 值选择 best_K
score = 0   # float('inf')  # 无穷大
best_K = 0
loss = []

# 降维后聚类 √ or 不降维聚类
# 设定不同 k 值进行聚类
print("开始对数据进行聚类...")
for K in range(5, 50, 1):
# for K in [13]:
    print("选取{}个聚类簇中心".format(K))
    cluster = KMeans(n_clusters=K, max_iter=1500)  # , verbose =1
    y = cluster.fit_predict(X_tsne).tolist()
    y = np.array(y)

    # 中心点
    cents = cluster.cluster_centers_  # 质心
    # 每个样本所属的簇
    labels = cluster.labels_
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    sse = cluster.inertia_
    # 从簇内的稠密程度和簇间的离散程度来评估聚类的效果 (越大越好)
    chs_values = calinski_harabasz_score(X_tsne, y)
    print("Calinski-Harabasz Score (越大越好):", chs_values)  # 112

    # loss.append(sse / K)
    loss.append(sum(np.min(cdist(X_tsne, cents, 'euclidean'), axis=1)) / X_tsne.shape[0])

    if chs_values >= score:
        score = chs_values
        best_K = K
        joblib.dump(cluster, kmeans_path)
        # writeClusterResult(y)   # 将聚类结果写入文件

# [手肘法]事后选择最佳 K 值
plt.plot(range(5, 50, 1), loss, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.savefig("result/手肘法最佳K值.png")
plt.show()

print("Calinski-Harabasz Score评估选择-最佳的簇个数：", best_K)   # 7



# print("#----------------------------------------#")
# print("#                                        #")
# print("#             3、降维数据可视化           #")
# print("#                                        #")
# print("#--------------------------------------#\n")

# visualization(kmeans_path, X_tsne)
