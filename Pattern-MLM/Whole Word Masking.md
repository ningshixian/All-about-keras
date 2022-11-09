## 简介

**Whole Word Masking (wwm)**，暂翻译为`全词Mask`或`整词Mask`，是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。 简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个子词，在生成训练样本时，这些被分开的子词会随机被mask。 在`全词Mask`中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask，即`全词Mask`。

**需要注意的是，这里的mask指的是广义的mask（替换成[MASK]；保持原词汇；随机替换成另外一个词），并非只局限于单词替换成`[MASK]`标签的情况。 更详细的说明及样例请参考：[#4](https://github.com/ymcui/Chinese-BERT-wwm/issues/4)**

同理，由于谷歌官方发布的`BERT-base, Chinese`中，中文是以**字**为粒度进行切分，没有考虑到传统NLP中的中文分词（CWS）。 我们将全词Mask的方法应用在了中文中，使用了中文维基百科（包括简体和繁体）进行训练，并且使用了[哈工大LTP](http://ltp.ai/)作为分词工具，即对组成同一个**词**的汉字全部进行Mask。

下述文本展示了`全词Mask`的生成样例。 **注意：为了方便理解，下述例子中只考虑替换成[MASK]标签的情况。**

| 说明         | 样例                                                         |
| ------------ | ------------------------------------------------------------ |
| 原始文本     | 使用语言模型来预测下一个词的probability。                    |
| 分词文本     | 使用 语言 模型 来 预测 下 一个 词 的 probability 。          |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |



## 模型对比

针对大家比较关心的一些模型细节进行汇总如下。

| -                 | BERTGoogle  | BERT-wwm               | BERT-wwm-ext         | RoBERTa-wwm-ext | RoBERTa-wwm-ext-large |
| ----------------- | ----------- | ---------------------- | -------------------- | --------------- | --------------------- |
| Masking           | WordPiece   | WWM[1]                 | WWM                  | WWM             | WWM                   |
| Type              | base        | base                   | base                 | base            | **large**             |
| Data Source       | wiki        | wiki                   | wiki+ext[2]          | wiki+ext        | wiki+ext              |
| Training Tokens # | 0.4B        | 0.4B                   | 5.4B                 | 5.4B            | 5.4B                  |
| Device            | TPU Pod v2  | TPU v3                 | TPU v3               | TPU v3          | **TPU Pod v3-32[3]**  |
| Training Steps    | ?           | 100KMAX128 +100KMAX512 | 1MMAX128 +400KMAX512 | 1MMAX512        | 2MMAX512              |
| Batch Size        | ?           | 2,560 / 384            | 2,560 / 384          | 384             | 512                   |
| Optimizer         | AdamW       | LAMB                   | LAMB                 | AdamW           | AdamW                 |
| Vocabulary        | 21,128      | ~BERT[4]               | ~BERT                | ~BERT           | ~BERT                 |
| Init Checkpoint   | Random Init | ~BERT                  | ~BERT                | ~BERT           | Random Init           |

**Q: 更多关于`RoBERTa-wwm-ext`模型的细节？**
A: 我们集成了RoBERTa和BERT-wwm的优点，对两者进行了一个自然的结合。 和之前本目录中的模型之间的区别如下:
1）预训练阶段采用wwm策略进行mask（但没有使用dynamic masking）
2）简单取消Next Sentence Prediction（NSP）loss
3）不再采用先max_len=128然后再max_len=512的训练模式，直接训练max_len=512
4）训练步数适当延长

需要注意的是，该模型并非原版RoBERTa模型，只是按照类似RoBERTa训练方式训练出的BERT模型，即RoBERTa-like BERT。 故在下游任务使用、模型转换时请按BERT的方式处理，而非RoBERTa。



## 使用建议

https://github.com/ymcui/Chinese-BERT-wwm/issues/39

<img src="/Users/ningshixian/Library/Application Support/typora-user-images/image-20210713174413955.png" alt="image-20210713174413955" style="zoom: 33%;" />

- **初始学习率是非常重要的一个参数（不论是`BERT`还是其他模型），需要根据目标任务进行调整。**

  在NLU排序任务中使用 `chinese_wwm_ext_L-12_H-768_A-12`：

  - *initial_learning_rate*=2e-5：train loss 趋于不变（lr偏小，可能收敛到**局部最优**解）;
  - *initial_learning_rate*=5e-5：train loss 正常收敛√；
  - *initial_learning_rate*=3e-4：train loss 趋于不变（lr偏大，无法收敛）；
  - *initial_learning_rate*=5e-5 + warmup=0.1 + weight_decay_rate=0.01：train loss 正常收敛，效果稍有提升√；



## [全词掩码训练的基本顺序](https://github.com/ymcui/Chinese-BERT-wwm/issues/13)

1. 对原始的句子进行中文切词（我们使用的是LTP，你也可以用别的做），得到seq_cws
2. 对原始句子进行WordPiece切词（BERT默认），得到seq_wp
3. 对seq_cws和seq_wp分析，得到字（wp）到词（cws）对应关系，即哪些连续的wordpiece属于一个中文词，这里中英文处理对应关系：

- `英文的一个词`对应`中文的一个词`
- `英文的一个WordPiece`对应`中文的一个字`

比如，单词`中华人民共和国`，切词后的结果是`中华 人民 共和国`，为了适配谷歌原版的wwm，你可以将其改为`中华 ##人民 ##共和国`，这样就能用谷歌原版的wwm处理了，当然这个只是为了识别字与词的从属关系，最终训练时需要把中文子词的 `##`前缀去掉（英文请保留，因为wordpiece处理过的英文是有可能包含`##`的）。



## [哈工大-中文BERT-wwm系列模型下载](https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)



## 参考

https://github.com/ymcui/Chinese-BERT-wwm

[修改Transformer结构，设计一个更快更好的MLM模型](https://kexue.fm/archives/7661)

