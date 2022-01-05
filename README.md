# 知识图谱实验说明

### 运行环境

本次实验我们使用了来自于 $THUNLP$ （清华大学自然语言处理实验室）开发的知识图谱表示学习包 $Open KE$ ，由于 $Open KE$ 对于Windows支持并不完善，我们使用WSL来完成此次实验。

具体环境为

* Windows 11
* WSL 2
* Ubuntu 18.04 LTS
* Anaconda 4.5.11
* CUDA 11.3
* PyTorch 1.10.0+cu113
* OpenKE PyTorch version

### 关键函数说明

本次实验我们通过调用OpenKE实现的多种知识图谱嵌入Translate模型，使用训练集进行训练，在测试集上进行预测工作。

我们调用了以下这些工具：

```python
import torch
import os
import hashlib
from OpenKE.openke.config import Trainer, Tester
from OpenKE.openke.module.model import TransE
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.strategy import NegativeSampling
from OpenKE.openke.data import TrainDataLoader, TestDataLoader
import numpy as np
from torch.autograd import Variable
import heapq
```

几个关键函数的定义如下

* `transform_the_raw_data()`：用于将原始数据集转化为OpenKE所支持的格式，以便调用OpenKE来进行处理和预测

  ```python
  def transform_the_raw_data():
      hash_obj = hashlib.md5()  # init the hash object
      entity_list = []
      entity_index_list = []
      relation_list = []
      train_list = []
      valid_list = []
      fake_test_list = []
      sup_ent = 0
      sup_rel = 0
  
      # deal with the train set
      with open(DATA_PATH+RAW_TRAIN, 'r', encoding='UTF-8') as f:
          line_list = f.readlines()
          for line in line_list:
              h, r, t = line.split('\t')
              if int(h) > sup_ent:
                  sup_ent = int(h)
              if int(t) > sup_ent:
                  sup_ent = int(t)
              if int(r) > sup_rel:
                  sup_rel = int(r)
              train_list.append((h, r, t))
  
      print('train set size:\t'+str(len(train_list)))
      print('fake test set size:\t'+str(len(fake_test_list)))  # here for local test
      print(f'max entity:{sup_ent}')
      print(f'max relation:{sup_rel}')
  
      with open(DATA_PATH+FORMAT_TRAIN, 'w', encoding='UTF-8') as f:  # rewrite the train set
          f.write(str(len(train_list))+'\n')
          for elem in train_list:
              f.write(elem[0]+' '+str(int(elem[2]))+' '+elem[1]+'\n')
  
      with open(DATA_PATH+FAKE_TEST, 'w', encoding='UTF-8') as f:  # rewrite the train set
          f.write(str(len(fake_test_list))+'\n')
          for elem in fake_test_list:
              f.write(elem[0]+' '+str(int(elem[2]))+' '+elem[1]+'\n')
  
      # deal with the valid set
      with open(DATA_PATH+RAW_VALID, 'r', encoding='UTF-8') as f:
          line_list = f.readlines()
          for line in line_list:
              h, r, t = line.split('\t')
              valid_list.append((h, r, t))
  
      print('valid set size:\t'+str(len(line_list)))
  
      with open(DATA_PATH+FORMAT_VALID, 'w', encoding='UTF-8') as f:  # rewrite the valid set
          f.write(str(len(valid_list))+'\n')
          for elem in valid_list:
              f.write(elem[0]+' '+str(int(elem[2]))+' '+elem[1]+'\n')
  
      # deal with the entity file
      with open(DATA_PATH+RAW_ENTITY_FILE, 'r',  encoding='UTF-8') as f:
          line_list = f.readlines()  # read a line
          for line in line_list:
              number, dscrpt = line.split('\t', 1)  # get the number of the entity
              hash_obj.update(dscrpt.encode('utf-8'))  # hash
              hashed_dscrpt = hash_obj.hexdigest()  # get the hashed value
              entity_index_list.append(int(number))
              entity_list.append((int(number), hex(int(hashed_dscrpt, 16))))  # push into the list
  
      hash_obj.update('0'.encode('UTF-8'))
      default_dscrpt = hash_obj.hexdigest()
      for index in range(0, sup_ent+1):
          if index not in entity_index_list:
              entity_list.append((index, hex(int(default_dscrpt, 16))))
  
      entity_list.sort(key=get_number_first)  # resort the list
  
      print('entity count:\t'+str(len(entity_list)))
  
      with open(DATA_PATH+FORMAT_ENTITY_FILE, 'w', encoding='UTF-8') as f:  # rewrite the entity file
          f.write(str(len(entity_list))+'\n')
          for elem in entity_list:
              f.write(elem[1]+'\t'+str(elem[0])+'\n')
  
      # deal with the relation file
      with open(DATA_PATH+RAW_RELATION_FILE, 'r',  encoding='UTF-8') as f:
          line_list = f.readlines()  # read lines
          for line in line_list:
              number, dscrpt = line.split('\t', 1)  # get the number of the relation
              hash_obj.update(dscrpt.encode('utf-8'))  # hash
              hashed_dscrpt = hash_obj.hexdigest()  # get the hashed value
              relation_list.append((int(number), hex(int(hashed_dscrpt, 16))))  # push into the list
  
      print('relation count:\t'+str(len(relation_list)))
      relation_list.sort(key=get_number_first)  # resort the list
  
      with open(DATA_PATH+FORMAT_RELATION_FILE, 'w', encoding='UTF-8') as f:  # rewrite the relation file
          f.write(str(len(relation_list))+'\n')
          for elem in relation_list:
              f.write(elem[1]+'\t'+str(elem[0])+'\n')
  ```

* `train_tanse()`：对TransE模型进行训练，并将训练结果存储于硬盘中：

  ```python
  def train_transe():
      # dataloader for training
      train_dataloader = TrainDataLoader(
          in_path=DATA_PATH,
          nbatches=100,
          threads=8,
          sampling_mode="normal",
          bern_flag=1,
          filter_flag=1,
          neg_ent=25,
          neg_rel=0)
  
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
  
      # train the model
      trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1000, alpha=1.0, use_gpu=True)
      trainer.run()
      transe.save_checkpoint('./checkpoint/transe.ckpt')
  ```

* `test_transe()`：对TransE模型进行性能测试：

  ```python
  def test():
      # dataloader for training
      train_dataloader = TrainDataLoader(
          in_path=DATA_PATH,
          nbatches=100,
          threads=8,
          sampling_mode="normal",
          bern_flag=1,
          filter_flag=1,
          neg_ent=25,
          neg_rel=0)
  
      # dataloader for test
      test_dataloader = TestDataLoader(DATA_PATH, "link", type_constrain=False)
  
      # define the model
      transe = TransE(
          ent_tot=train_dataloader.get_ent_tot(),
          rel_tot=train_dataloader.get_rel_tot(),
          dim=200,
          p_norm=1,
          norm_flag=True)
  
      # test the model
      transe.load_checkpoint('./checkpoint/transe.ckpt')
      tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
      tester.run_link_prediction(type_constrain=False)
  ```

* `predict_transe()`：调用TransE模型进行结果预测：

  ```python
  def predict_transe():
      # dataloader
      train_dataloader = TrainDataLoader(
          in_path=DATA_PATH,
          nbatches=100,
          threads=8,
          sampling_mode="normal",
          bern_flag=1,
          filter_flag=1,
          neg_ent=25,
          neg_rel=0)
  
      # define the model
      transe = TransE(
          ent_tot=train_dataloader.get_ent_tot(),
          rel_tot=train_dataloader.get_rel_tot(),
          dim=200,
          p_norm=1,
          norm_flag=True)
  
      query_list = []
      answer_list = []
      test_dict = {}
      with open(DATA_PATH+TEST, 'r', encoding='UTF-8') as f:
          line_list = f.readlines()
          for line in line_list:
              h, r, q = line.split()
              query_list.append((h, r))
  
      ent_array = np.array(range(0, train_dataloader.entTotal))
      print(ent_array)
  
      i = 0
      for elem in query_list:
          test_dict['batch_h'] = np.array([int(elem[0])])
          test_dict['batch_r'] = np.array([int(elem[1])])
          test_dict['batch_t'] = ent_array
          test_dict['mode'] = 'tail_batch'
          answer = transe.predict({
              'batch_h': Variable(torch.from_numpy(test_dict['batch_h'])),
              'batch_r': Variable(torch.from_numpy(test_dict['batch_r'])),
              'batch_t': Variable(torch.from_numpy(test_dict['batch_t'])),
              'mode': test_dict['mode']
          })
          single_answer_list = list(map(list(answer).index, heapq.nlargest(5, list(answer))))
          answer_list.append(single_answer_list)
          print(i)
          i += 1
  
      with open(DATA_PATH+ANSWER, 'w', encoding='UTF-8') as f:  # rewrite the relation file
          for elem in answer_list:
              f.write(str(elem[0])+','+str(elem[1])+','+str(elem[2])+','+str(elem[3])+','+str(elem[4])+'\n')
  
      print('Done!')
  ```

  

