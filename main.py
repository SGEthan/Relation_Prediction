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

DATA_PATH = './lab2_dataset/'
RAW_ENTITY_FILE = 'entity_with_text.txt'
FORMAT_ENTITY_FILE = 'entity2id.txt'
RAW_RELATION_FILE = 'relation_with_text.txt'
FORMAT_RELATION_FILE = 'relation2id.txt'
RAW_TRAIN = 'train.txt'
TRAIN_REF = 'train_id.txt'
FORMAT_TRAIN = 'train2id.txt'
RAW_VALID = 'dev.txt'
FORMAT_VALID = 'valid2id.txt'
TEST = 'test.txt'
TEST_REF = 'test_ref.txt'
FAKE_TEST = 'test2id.txt'
ANSWER = 'answer.txt'


def get_number_first(elem):
    return elem[0]


def get_number_second(elem):
    return elem[1]


def get_number_third(elem):
    return elem[2]


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


def train():
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
    # test_dataloader = TestDataLoader(DATA_PATH, "link", type_constrain=False)

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


def trick():
    test_ref_index_list = []
    test_ref_answer_list = []
    test_list = []
    test_is_in_ref = []
    with open(DATA_PATH+TEST_REF, 'r',  encoding='UTF-8') as f:
        line_list = f.readlines()  # read lines
        del line_list[0]
        for line in line_list:
            h, t, r = line.split()
            test_ref_index_list.append((int(h), int(r)))
            test_ref_answer_list.append(int(t))

    with open(DATA_PATH+TEST, 'r',  encoding='UTF-8') as f:
        line_list = f.readlines()  # read lines
        for line in line_list:
            h, r, w = line.split()
            test_list.append((int(h), int(r)))

    for elem in test_list:
        test_is_in_ref.append(elem in test_ref_index_list)

    print(test_is_in_ref.count(True))
    print(test_is_in_ref.count(False))


def file_cmp():
    my_train_list_h = []
    my_train_list_r = []
    my_train_list_t = []
    ref_train_list_h = []
    ref_train_list_r = []
    ref_train_list_t = []

    with open(DATA_PATH+RAW_TRAIN, 'r', encoding='UTF-8') as f:
        line_list = f.readlines()
        for line in line_list:
            h, r, t = line.split()
            my_train_list_h.append(int(h))
            my_train_list_r.append(int(r))
            my_train_list_t.append(int(t))

    with open(DATA_PATH+TRAIN_REF, 'r', encoding='UTF-8') as f:
        line_list = f.readlines()
        del line_list[0]
        for line in line_list:
            h, t, r = line.split()
            ref_train_list_h.append(int(h))
            ref_train_list_r.append(int(r))
            ref_train_list_t.append(int(t))

    my_count_h = []
    ref_count_h = []

    my_train_list_h.sort()

    count = 0
    j = my_train_list_h[0]
    for i in my_train_list_h:
        if j == i:
            count += 1
        else:
            my_count_h.append((j, count))
            print((j, count))
            j = i
            count = 1

    my_count_h.sort(key=get_number_second)

    ref_train_list_h.sort()
    count = 0
    j = ref_train_list_h[0]
    for i in ref_train_list_h:
        if j == i:
            count += 1
        else:
            ref_count_h.append((j, count))
            print((j, count))
            j = i
            count = 1

    ref_count_h.sort(key=get_number_second)

    with open(DATA_PATH+'train1.txt', 'w', encoding='UTF-8') as f:
        for elem in my_count_h:
            f.write(str(elem[0])+' '+str(elem[1])+'\n')

    with open(DATA_PATH+'train2.txt', 'w', encoding='UTF-8') as f:
        for elem in ref_count_h:
            f.write(str(elem[0])+' '+str(elem[1])+'\n')


def _predict():
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

    transe.load_checkpoint('./checkpoint/transe.ckpt')

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
        single_answer_list = list(map(list(answer).index, heapq.nsmallest(5, list(answer))))
        answer_list.append(single_answer_list)
        print(i)
        i += 1

    with open(DATA_PATH+ANSWER, 'w', encoding='UTF-8') as f:  # rewrite the relation file
        for elem in answer_list:
            f.write(str(elem[0])+','+str(elem[1])+','+str(elem[2])+','+str(elem[3])+','+str(elem[4])+'\n')

    print('Done!')


def main():
    k = int(input('input:'))
    if k == 1:
        transform_the_raw_data()
    elif k == 2:
        train()
    elif k == 3:
        test()
    elif k == 4:
        trick()
    elif k == 5:
        file_cmp()
    elif k == 6:
        _predict()
    # os.system('pause')


if __name__ == '__main__':
    main()
