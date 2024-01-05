# Learning Stable Classifiers by Transferring Unstable Features 方法

> 参考： https://zhuanlan.zhihu.com/p/581626707


## Define

- 不同环境：即拥有不同不稳定特征分布的不同数据集

## 1. Identify spurious correlations from the source tasks

- 输入： `N` 个 不同环境 （TOFU的MNIST数据集只给了2个环境）
- 输出： `N(N-1) * M * 2` 个 数据集 
  - `N` : N 个环境
  - `N-1` : 在其它环境训练的分类器的个数
  - `M` : 分类任务的类别数
  - `2` : 为分类错误或分类正确
- 例子 ( `N=2`)： 
    ```
    ENV_e0_LABEL_y0_correct
    ENV_e0_LABEL_y0_mistake
    ENV_e0_LABEL_y1_correct
    ENV_e0_LABEL_y1_mistake
    ENV_e1_LABEL_y0_correct
    ENV_e1_LABEL_y0_mistake
    ENV_e1_LABEL_y1_correct
    ENV_e1_LABEL_y1_mistake
    ```
- 流程
  1. `N` 个 环境上分别训练 `N` 个分类器模型
  2. 对每个环境，收集其它分类器在该环境上的分类结果，根据每个标签、分类正确与否，划分为新的数据集

- 伪代码
  ``` Python
  N envs, N models
  train N models on N envs
  for i in range(N):
    env = envs[i]
    for j in range(N):
      if i == j:
        continue
      model = models[j]
      predict = model.predict(env.X)
      label = env.y
      correct = predict == label
      correct_env = env.X[correct]
      mistake_env = env.X[~correct]
      for label in range(M):
        correct_label_env = correct_env[correct_env.y == label]
        mistake_label_env = mistake_env[mistake_env.y == label]
  ```
- note: 之所以要按 label划分数据集，是为让后面的训练采样到的每个类别数量一致

## 2. Learn an unstable feature representation

- 输入： `N(N-1) * M * 2` 个 数据集，一个 模型，用于表征不稳定特征
  - `N` : N 个环境
  - `N-1` : 在其它环境训练的分类器的个数
  - `M` : 分类任务的类别数
  - `2` : 为分类错误或分类正确
- 输出： 不稳定特征的模型参数，即通过上述数据训练一个表征不稳定特征的模型
  - 本模型类似一个编码器，编码数据的不稳定特征。（不知道为什么，TOFU中，本模型采用了和分类器一模一样的 CNN + MLP）
  - 但是模型有 embedding 的 CNN 和 clf 的 MLP 两部分，这里训练的是 embedding 的 CNN 部分
  - TOFU中 CNN的 embedding 部分的输出是 300维
  
- 公式：
    $$
f_Z = \arg \min \sum_{y, E_i \neq E_j} \mathbb{E}_{X_1^\checkmark,X_2^\checkmark,X_3^\times} [L_Z(X_1^\checkmark,X_2^\checkmark,X_3^\times)] \\ 
L_Z(X_1^\checkmark, X_2^\checkmark,X_3^\times) = \max (0, \delta+ \| \overline{f_Z}(X_1^\checkmark) -\overline{f_Z}(X_2^\checkmark) \| _2^2 - \| \overline{f_Z}(X_1^\checkmark) -\overline{f_Z}(X_3^\times) \| _2^2 \\
    $$

- 流程：
  1. 将上述 `N(N-1) * M * 2` 个数据集 分为一对对数据集 `DS_Correct` 和 `DS_Mistake` （一共 N(N-1) * M 对）
     - 论文里为了方便表达公式，有三个数据集 `DS_Correct_1`, `DS_Correct_2`, `DS_Mistake`
     - 但在实际代码中，`DS_Correct_1` 和 `DS_Correct_2` 是同一个数据集 `DS_Correct` 的 
  2. 对于每一对数据集`DS_Correct` 和 `DS_Mistake` ，对模型的嵌入层，这里是CNN，进行训练
     - 注意，这里的训练是同时进行的，即 先遍历batch，再在内部 遍历 数据集对
  3. 这里为什么能得到不稳定特征的表示呢？
     - 基于一个 insight ：划分对的数据大概率更关注真正的特征，而划分错的数据大概率关注了不稳定特征
     - 所以对于每个分类，
       - 让划分对的数据的嵌入层之间尽可能接近：这里损失函数定义为 `L2_norm(x_pos_1 - x_pos_2)`
       - 而划分错的数据和划分对的数据的嵌入层之间尽可能远离: 这里损失函数定义为 `- L2_norm(x_pos - x_neg)`
  
- 训练代码 (详细代码参考 https://github.com/YujiaBao/tofu/blob/main/src/tofu/partition.py)
```Python
for epoch in range(args.epoch):
    for batch_id in range(args.batch_num):
        for DS_Correct, DS_Mistake in pairs:
            x_pos   = DS_Correct.get_batch(batch_id)
            x_neg   = DS_Mistake.get_batch(batch_id)

            ebd_pos = model['ebd'](x_pos)
            ebd_neg = model['ebd'](x_neg)

            # 这里解释一下 L2 的计算，这里假设 batch_size = n, ebd_dim = d
            # 那么 ebd_pos 的 shape 为 (n, d)
            # 这里对 batch内部的每对数据计算 L2，所以得到的 shape 为 (n, n)
            #    具体计算为：
            #       1. ebd_pos.unsqueeze(1) , 得 ebd_pos1 的 shape 为 (n, 1, d)
            #       2. ebd_pos.unsqueeze(0) , 得 ebd_pos2 的 shape 为 (1, n, d)
            #       3. diff_pos_pos = ebd_pos1 - ebd_pos2 , 得 diff_pos_pos 的 shape 为 (n, n, d)
            #       4. L2 = torch.norm(diff_pos_pos, dim=2) , 得 diff_pos_pos 的 shape 为 (n, n)
            diff_pos_pos = compute_l2(ebd_pos, ebd_pos)
            diff_pos_neg = compute_l2(ebd_pos, ebd_neg)

            loss = (
                torch.mean(torch.max(torch.zeros_like(diff_pos_pos),
                                    diff_pos_pos - diff_pos_neg +
                                    torch.ones_like(diff_pos_pos) *
                                        args.thres)))
            loss.backward()
```

## 3. Transfer the unstable feature to the target task
- 输入： 一个embed不稳定特征的模型 partition_model（前面训练好的），一个目标数据集 (train + valid)
- 输出： 一个excellent的分类器模型 `f`
- 流程：
  1. 对每个标签 y 对应的数据集 Xs，用 partition_model 得到每个X的嵌入表达，进行 K-Means 聚类，最终聚类结果 
     - 假设有 `M` 个标签，`K` 聚类，那么最终得到 `M*K` 个聚类
  2. 使用上面聚类好的 `M*K` 个聚类，进行训练, 
     - 在每个batch内，有 `M*K` 个loss （每个聚类一个），只使用表现最差的 loss进行反向传播

- 公式
  
$$
f = \arg \min \max_{i,y} L(f(C_i^y)) \\
$$

- 训练代码: (详细代码参考  https://github.com/YujiaBao/tofu/blob/main/src/tofu/utils.py#L41)
```Python
for epoch in range(args.epoch):
    # 一共 M*K 个聚类, 所以一共有 M*K 个 batch
    for batches in zip(*train_loaders):
        # work on each batch
        model['ebd'].train()
        model['clf'].train()

        x, y = [], []

        for batch in batches:
            batch = to_cuda(squeeze_batch(batch))
            x.append(batch['X'])
            y.append(batch['Y'])

        # M*K 个聚类的预测结果一起并行计算
        pred = model['clf'](model['ebd'](torch.cat(x, dim=0)))

        cur_idx = 0

        # 计算 M*K 个聚类的 loss
        for cur_true in y:
            cur_pred = pred[cur_idx:cur_idx+len(cur_true)]
            cur_idx += len(cur_true)

            loss = F.cross_entropy(cur_pred, cur_true)

            if loss.item() > worst_loss:
                worst_loss = loss
                worst_acc = acc

        opt.zero_grad()
        worst_loss.backward()
        opt.step()
```