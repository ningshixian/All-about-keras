**Constellation Loss: Improving the efficiency of deep metric learning loss functions for optimal embedding.**

Metric learning has become an attractive field for research on the latest years. Loss functions like contrastive loss , triplet loss  or multi-class N-pair loss have made possible generating models capable of tackling complex scenarios with the presence of many classes and scarcity on the number of images per class not only work to build classifiers, but to many other applications where measuring similarity is the key. Deep Neural Networks trained via metric learning also offer the possibility to solve few-shot learning problems. Currently used state of the art loss functions such as triplet and contrastive loss functions, still suffer from slow convergence due to the selection of effective training  samples that has been partially solved by the multi-class N-pair loss by simultaneously adding additional samples from the different classes. In this work, we extend triplet and multiclass-N-pair loss function by proposing the *constellation loss* metric where the distances among all class combinations are simultaneously learned. We have compared our *constellation loss* for visual class embedding showing that our loss function over-performs the other methods by obtaining more compact clusters while achieving  better classification results.

This study has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 732111 [PICCOLO project](https://www.piccolo-project.eu/)


This [code repository](https://git.code.tecnalia.com/comvis_public/piccolo/constellation_loss/-/tree/master/) corresponds to the work submitted to the Thirty-third Conference on Neural Information Processing Systems (NeurIPS 2019).

Arxiv version of the paper can be found at [link](http://arxiv.org/abs/1905.10675). Please, cite this paper if you use this code.



