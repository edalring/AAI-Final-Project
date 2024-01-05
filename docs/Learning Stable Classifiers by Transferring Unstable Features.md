> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/581626707)

陆续分享一些 Invariant Learning 相关的文章，这篇与 之前看的 IRM 不太一样。

[Learning Stable Classifiers by Transferring Unstable Features](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/2106.07847)

Motivation
----------

1.  如果只提供（input-label）对，算法可能没有足够的信息来区分_稳定 (causal) 特征_和_不稳定 (spurious) 特征_。
2.  虽然源分类器在做出最终预测时没有偏见，但其内部的连续 representation 仍然可以编码关于不稳定特征的信息。
3.  通过 re-using 或 fine-tuning 在**源任务**上学习到的 representation 进行的_直接 transfer_ 到**目标任务**中失败，性能不优于多数 baseline。
4.  **问题：**如何利用源数据环境计算不稳定_(spurious)_的特征表示？
5.  在具有相同标签值的示例中，具有**相同预测结果的示例**比**具有不同预测结果的示例**具有**更相似的不稳定特性** (Theroem 1)。直觉上，如果环境 $E_i$E_i 中不稳定的相关性更强，那么当这些更强的相关性不成立时，分类器 $f_i$ f_i 会过度使用这些相关性，并在 $E_j$ E_j 上出错。_（就是说，源任务中那些分对的样本比分错的更能让分类器学习到与标签相关的虚假表征）_

![](https://pic3.zhimg.com/v2-ba20b8d9d39a57171fd0126689f6d0ca_r.jpg)

Method- 算法 TOFU （Transfer OF Unstable features）
-----------------------------------------------

![](https://pic4.zhimg.com/v2-d6ad6fc9637a3b59f7ce7016b2828f57_r.jpg)

### 1. 从源任务推断不稳定的特征

在源任务的环境（ ，$E_1，E_2, \cdots$E_1，E_2, \cdots ）中识别出不稳定特征 $Z(x)$ Z(x) （unstable feature），通过 metric learning 通过 $f_Z(x)$f_Z(x) 为 $Z(x)$Z(x) 编码。

具体细节如下：

*   **（S1）**为每个环境 $E_i$E_i 单独训练分类器 $f_i$ f_i ；
*   **（S2）**为每对环境 $E_i$ E_i 和 $E_j$E_j ，用分类器 $f_i$ f_i 将 $E_j$E_j 中的样本分为两类： $E_j^\checkmark$E_j^\checkmark 和 $E_j^\times$ E_j^\times （分对，分错）；
*   **（S3）**通过在所有环境对 $E_j, E_i$E_j, E_i 和所有可能的标签值 $y$y 上最小化方程来学习一个不稳定的特征表示 $f_Z$ f_Z ：

$$f_Z = \arg \min \sum_{y, E_i \neq E_j} \mathbb{E}_{X_1^\checkmark,X_2^\checkmark,X_3^\times} [L_Z(X_1^\checkmark,X_2^\checkmark,X_3^\times)] \\ L_Z(X_1^\checkmark, X_2^\checkmark,X_3^\times) = \max (0, \delta+ \| \overline{f_Z}(X_1^\checkmark) -\overline{f_Z}(X_2^\checkmark) \| _2^2 - \| \overline{f_Z}(X_1^\checkmark) -\overline{f_Z}(X_3^\times) \| _2^2\\$$

f_Z = \arg \min \sum_{y, E_i \neq E_j} \mathbb{E}_{X_1^\checkmark,X_2^\checkmark,X_3^\times} [L_Z(X_1^\checkmark,X_2^\checkmark,X_3^\times)] \\ L_Z(X_1^\checkmark, X_2^\checkmark,X_3^\times) = \max (0, \delta+ \| \overline{f_Z}(X_1^\checkmark) -\overline{f_Z}(X_2^\checkmark) \| _2^2 - \| \overline{f_Z}(X_1^\checkmark) -\overline{f_Z}(X_3^\times) \| _2^2\\

其中 $X_1^\checkmark, X_2^\checkmark$X_1^\checkmark, X_2^\checkmark 是从 $E_j^{i\checkmark}|_y$E_j^{i\checkmark}|_y 中随机采样的， $E_j^{i\times}|_y$E_j^{i\times}|_y 中随机采样的。

通过优化上式，我们鼓励具有类似不稳定特征的 sample 在表示 $f_Z$ f_Z 上接近。

**个人思考：**

*   但是这个损失函数设计的我有点没懂，找了下参考文献 [[1]](#ref_1)，发现应该是出自 Passive Aggressive Algorithms 一类，其原本目的主要是增量学习，并据以上的判断准则优化自己的模型：

![](https://pic2.zhimg.com/v2-1a114ee7f5760edeed39d7c24277f825_r.jpg)

1.  在样本分类正确 且 模型对可能性的预测准确（程度大于 $\delta$ \delta ）时，模型不做调整（这里体现出了被动）
2.  在样本分类正确 但 模型对可能性的预测有失偏颇（不太准确）时，模型做出轻微的调整
3.  在样本分类错误时，模型做出较大的调整（体现出较强的 “攻击性”)

在这里我们调整的目的是为了让 $f_Z$ f_Z 学习到的是与目标相关的虚假特征，找出不稳定特征的样本（Passive），用损失取代学习率，并直接用样本优化参数（Aggressive）。

### 2. 为目标任务学习稳定的相关性

给定不稳定的特征表示 $f_Z$ f_Z ，我们的目标是学习一个目标分类器，它关注稳定的相关性，而不是使用不稳定的特征。

在 DRO[[2]](#ref_2) 的启发下，我们最小化代表不同不稳定特征值的一组实例的最坏情况风险。然而，与 DRO 不同，这些组是基于先前学习的表示 $f_Z$f_Z 自动构建的。

*   （T1）根据标签 $y$y ，用 $f_Z$ f_Z 为进行 K-Means 聚类，最终聚类结果 $C_1^y, \cdots, C_{n_c}^y$C_1^y, \cdots, C_{n_c}^y ；
*   （T2）通过最小化所有 cluster 上的_最坏情况_风险来训练目标分类器 $f$f ：

$$f = \arg \min \max_{i,y} L(C_i^y) \\$$

f = \arg \min \max_{i,y} L(C_i^y) \\

Discussion
----------

作者还在附录进行了讨论并做了简要回答，感兴趣的可以去看看原文

*   Are biases shared across real-world tasks?
*   What if the source task and target task are from different domains?
*   Can we apply domain-invariant representation learning (DIRL) directly to the source environments?
*   What if the mistakes correspond to other factors such as label noise, distribution shifts, etc.?
*   Is the algorithm efficient when multiple source environments are available?
*   Why does the baselines perform so poorly on MNIST?
*   Why do the baselines behave so differently across different datasets?
*   How many clusters to generate?
*   How do we select the hyper-parameter for TOFU?
*   Ablation study

（整体工作量真的很大，不愧是 ICLR，但是理论推导到具体损失函数的部分稍稍有点突兀，感觉解释得不是很清楚，我对于为什么这么设计就可以达到学习不稳定表征的原因没有理解很到位。）

参考
--

1.  [^](#ref_1_0)Chechik, G., Sharma, V., Shalit, U., & Bengio, S. (2010). Large Scale Online Learning of Image Similarity Through Ranking. Journal of Machine Learning Research, 11(3). [https://www.jmlr.org/papers/volume11/chechik10a/chechik10a.pdf](https://www.jmlr.org/papers/volume11/chechik10a/chechik10a.pdf)
2.  [^](#ref_2_0)Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P. Distributionally robust neural networks. In International Conference on Learning Representations, 2020.  [https://openreview.net/forum?id=ryxGuJrFvS](https://openreview.net/forum?id=ryxGuJrFvS)