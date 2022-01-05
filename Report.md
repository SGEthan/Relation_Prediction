# Web信息处理与应用第二次实验报告

### 算法概述

这次实验中，我们使用了知识表示学习的模型。算法的大致原理如下：

#### TransE模型

$Translating\;Embedding$ ，简称为 $TransE$ ，是一种用于知识表示学习的模型，发表于论文[Translating Embeddings for Modeling Multi-relational Data (neurips.cc)](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)

其基本原理，由于 $word2vec$ 向量具有平移不变性，我们可以考虑这样一种思想：将头尾实体节点分别视作向量，关系视作同一个空间的向量，它表示从头向量到尾向量的差值，即我们有这样的关系：对于三元组 $(h,r,t)$，有
$$
\mathbf{h}+\mathbf{l}\approx\mathbf{t}
$$
如此，我们需要学习的就是每个头尾实体以及关系实体在向量空间中的表示，我们将知识图谱中的实体和关系分别存于两个矩阵，实体矩阵为 $n\times d$，每一行为一个实体对应的 $d$ 维向量；关系矩阵同理。在使用模型进行预测时（以预测尾节点为例），我们计算头节点和关系节点的和，得到一个结果，并在所有实体向量中寻找与之距离最相近的 $top\;k$ 个，作为预测结果输出。

$TransE$ 模型的 $loss\;function$ 是这样设计的：
$$
\mathcal{L}=\sum_{(h,l,t)\in S}\sum_{(h',l,t')\in S'}[\gamma+d(\symbfit{h}+\symbfit{l,\symbfit{t}})-d(\symbfit{h'}+\symbfit{l,\symbfit{t'}})]_+
$$
其中
$$
[x]_+=max(0,x)
$$
这里采用了 $Negative\; Sampling$ 的方法，即，相对反例而言，给予正例更高的分数。式中的 $(h',l,t')$ 是随机替换掉头节点或尾节点的到的反例，$\gamma$ 是 $Margin$，即用于修正正反例之间的误差的量。

论文中给出了详细的算法流程：

<img src="https://img-blog.csdnimg.cn/20190504224500785.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NodW5hb3hpMjMxMw==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:67%;" />

### 实验过程记录

#### 对于OpenKE的探索

本次实验中，我们调用了 $THUNLP$ （清华大学自然语言处理实验室）开发的知识图谱表示学习包 $Open KE$ ，它很好的实现了上述的知识表示学习模型，并且提供了 $GPU$ 支持，性能有所保证。

首先我们研究了一下 $OpenKE$ 官方所给出的examples，了解了 $OpenKE$ 的workflow如下：

* 进行数据的载入，分别导入训练集和测试集，这里 $OpenKE$ 对于数据进行了良好的封装，有 `TrainDataLoader` 以及`TestDataLoader`类方便处理：

  ```python
  # dataloader for training
  train_dataloader = TrainDataLoader(
  	in_path="./benchmarks/FB15K237/",
  	nbatches=100,
  	threads=8,
  	sampling_mode="normal",
  	bern_flag=1,
  	filter_flag=1,
  	neg_ent=25,
  	neg_rel=0)
  
  # dataloader for test
  test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link", type_constrain=False)
  ```

* 定义模型以及 $loss\;funtion$ （这里以 $TransE$  为例）：

  ```python
  # define the model
  transe = TransE(
  	ent_tot=train_dataloader.get_ent_tot(),
  	rel_tot=train_dataloader.get_rel_tot(),
  	dim=200,
  	p_norm=1,
  	norm_flag=True)
  
  
  # define the loss function
  model = NegativeSampling(
  	model=transe,
  	loss=MarginLoss(margin=5.0),
  	batch_size=train_dataloader.get_batch_size()
  )
  ```

* 进行模型的训练，以及checkpoint的保存：

  ```python
  # train the model
  trainer = Trainer(model=model, data_loader=train_dataloader, train_times=10, alpha=1.0, use_gpu=True)
  trainer.run()
  transe.save_checkpoint('./checkpoint/transe.ckpt')
  ```

* 进行模型的性能测试：

  ```python
  # test the model
  transe.load_checkpoint('./checkpoint/transe.ckpt')
  tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
  tester.run_link_prediction(type_constrain=False)
  ```

由于 $OpenKE$ 包良好的封装，我们可以很简单的对于模型进行训练以及性能测试。但同样由于其不算完善的$API$ ，以及相当糟糕的文档编写和版本管理（新版本和老版本的workflow有着较大区别），我们不能很显然的了解到如何使用其完成链接预测任务。于是我们对于 $OpenKE$ 的源代码又进行了一些探索，但因为 $OpenKE$ 的代码并没有任何注释，整个过程相当艰难，好在其变量和函数名相对直观，以及代码组织不算特别复杂，这个过程还是进行了下去。

从模型本身入手，以 $TransE$ 为例，我们可以看到模型内部有这样的方法定义：

```python
def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()
```

可以看出这个方法的作用是进行预测，并输出结果。但方法本身对于输入数据的格式和输出数据的含义并没有任何注释描述，我们只能在别的文件里面寻找这个函数的引用，以及如何使用它。

从测试部分入手，我们可以看到，在测试部分，有调用`predict`方法的痕迹：

```python
def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
```

以及上述代码中提到的 `to_var` 函数：

```python
def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))
```

由此我们可以得到，`predict`方法接收的输入是一个字典，其中三项值为`Tensor`类型，一项为字符串，用于指示模式。四项数据具体格式的探索就不展开了（这段过程也花费了较大时间和精力），其格式如下（以尾节点预测任务为例）：

* `batch_h`：一个 $1\times1$ 的 Tensor，是需要预测的样本头节点编号
* `batch_t`：一个 $n\times1$ 的 Tensor， $n$ 是实体数
* `batch_r`：一个 $1\times1$ 的 Tensor，是需要预测的样本关系节点编号
* `mode`：对于尾节点预测，其值为`'tail_batch'`

由此，我们对于使用 $OpenKE$ 包进行知识表示学习模型的训练，测试，使用，都有了足够的了解，可以进行后面的工作了。

#### 我们的工作

我们按照如前所述的知识，制定工作流，并执行实验要求的预测任务：

* 对于原始数据集进行处理和重新包装：

  考虑到我们被提供的数据集和 $OpenKE$ 标准的数据集格式有所区别，必须先进行预处理，我们定义了`transform_the_raw_data()`函数来进行数据集格式的转换

* 模型的训练：

  我们定义了这样一个函数`train()`，调用 $OpenKE$ 来训练我们指定的模型，并将训练结果保存于`./checkpoints/`中

* 模型的测试：

  前几轮训练时，我们的训练集为全部训练集的其中一部分，我们按照9：1的比例随机采集出其中的10%作为线下测试集来进行测试，我们定义了这样一个函数`test()`，调用 $OpenKE$ 来测试我们训练好的模型，测试结果示例如下：

  ```
  metric:                  MRR             MR              hit@10          hit@3           hit@1
  l(raw):                  0.262440        233.330063      0.520829        0.311698        0.139421
  r(raw):                  0.402787        25.889549       0.699853        0.467437        0.260359
  averaged(raw):           0.332614        129.609802      0.610341        0.389567        0.199890
  
  l(filter):               0.443760        98.402710       0.687202        0.539127        0.302824
  r(filter):               0.606719        9.766116        0.867583        0.715878        0.454602
  averaged(filter):        0.525239        54.084412       0.777393        0.627503        0.378713
  0.777393
  0.7773927450180054
  ```

* 完成预测任务：

  我们定义了这样的一个函数`_predict()`，调用 $OpenKE$ ，使用我们训练好的模型，得到预测结果，并保存于`./lab2_dataset/answer.txt`中，用于在线提交预测结果。

