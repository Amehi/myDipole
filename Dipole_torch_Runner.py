# -*- coding: utf-8 -*-
# @Time    : 2019-07-09 16:05
# @Author  : Yuyoo
# @Email   : sunyuyaoseu@163.com
# @File    : Dipole_torch_Runner.py

import sys
import numpy as np
import time
from dipole_torch.Dipole_torch import Dipole
import math
from operator import itemgetter
from utils import metric_report
import torch.nn as nn
from torch.optim import Adadelta
import torch
from torch.utils import data
import pickle
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score
import random
import argparse
import json
from pathlib import Path
from mydataloader import TrainDataset, EvalDataset
import torch.utils.data as data_utils
from tqdm import tqdm
from Dipole_test import ndcg_at_k, precision_at_k
starttime = time.time()

def multi_label_classification_task_eval(pred, targets, threshold=0.5):
    pred_binary = (pred > threshold).astype(int)

    # 计算F1分数 (micro, macro, and weighted)
    f1_micro = f1_score(targets, pred_binary, average='micro')
    f1_macro = f1_score(targets, pred_binary, average='macro')
    # f1_weighted = f1_score(targets, pred_binary, average='weighted')

    # 计算准确率
    accuracy = accuracy_score(targets, pred_binary)

    # 计算召回率
    recall_micro = recall_score(targets, pred_binary, average='micro')
    recall_macro = recall_score(targets, pred_binary, average='macro')
    # recall_weighted = recall_score(targets, pred_binary, average='weighted')

    # 计算AUC-ROC（注意：必须是概率或原始分数作为输入）
    auc_roc = roc_auc_score(targets, pred, multi_class='ovr')
    # 计算AUC-PR (平均精度)
    auc_pr = average_precision_score(targets, pred)
    return {
        "acc": accuracy,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro
    }

def process_data(dataset_name):
    dataset_path = Path(f"processed_data/dataset_{dataset_name}.pkl")
    if dataset_path.is_file():
        print('Already preprocessed. Skip preprocessing')
        return
    if not dataset_path.parent.is_dir():
        dataset_path.parent.mkdir(parents=True)
    with open(f'processed_data/{dataset_name}/trajs.pkl', 'rb') as f:
        trajs = pickle.load(f)
    with open(f'processed_data/{dataset_name}/gds.pkl', 'rb') as f:
        gds = pickle.load(f)
    with open(f'processed_data/{dataset_name}/action_dict.json', 'r') as f:
        action_dict = json.load(f)
    with open(f'processed_data/{dataset_name}/action_dict_l.json', 'r') as f:
        action_dict_l = json.load(f)
    random.seed(2024)
    n = len(trajs)
    indices = list(range(n))
    random.shuffle(indices)
    trajs = [trajs[i] for i in indices]
    gds = [gds[i] for i in indices]
    # times = [times[i] for i in indices]

    #split
    trajs_train = trajs[ : int(0.75 * n)]
    gds_train = gds[: int(0.75 * n)]
    # times_train = times[: int(0.75 * n)]
    trajs_val = trajs[int(0.75 * n) : int(0.85 * n)]
    gds_val = gds[int(0.75 * n) : int(0.85 * n)]
    # times_val = times[int(0.75 * n) : int(0.85 * n)]
    trajs_test = trajs[int(0.85 * n) : ]
    gds_test = gds[int(0.85 * n) : ]
    # times_test = times[int(0.85 * n) : ]

    
    #store
    dataset = {
        'train':(trajs_train, gds_train),
        'val' : (trajs_val, gds_val),
        'test' : (trajs_test, gds_test),
        'smap' : action_dict,
        'smap_l' : action_dict_l,
    }
    with dataset_path.open('wb') as f:
        pickle.dump(dataset, f)

def load_data(dataset_name):
    if dataset_name == 'mimic4':
        dataset_path = Path("processed_data/dataset_mimic4.pkl")
        process_data('mimic4')
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset
    elif dataset_name == 'mimic3':
        dataset_path = Path("processed_data/dataset_mimic3.pkl")
        process_data('mimic3')
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset
    else:
        print("error matching dataset name")




def fix_random_seed_as(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True


def train_predict(args):
    batch_size=100
    epochs=100
    topk=30
    L2=1e-8
    fix_random_seed_as(args.seed)
    device = torch.device("cuda:4" if torch.cuda.is_available() == True else 'cpu')


    data = load_data(args.dataset)
    train_data = data['train']
    dev_data = data['val']
    test_data = data['test']
    smap_l = data['smap_l']

    train_dataset = TrainDataset(train_data, smap_l)
    val_dataset = EvalDataset(dev_data, smap_l)
    test_dataset = EvalDataset(test_data, smap_l)
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = Dipole(input_dim=1, day_dim=100, rnn_hiddendim=300, output_dim=len(data['smap_l'])+1)


    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))

    optimizer = Adadelta(model.parameters(), lr=1, weight_decay=L2)
    loss_mce = nn.CrossEntropyLoss(reduction='sum')
    model = model.to(device)
    if args.resume:
        checkpoint = torch.load(f"./model/saved_model_seed{args.seed}_{args.dataset}")
        save_epoch = checkpoint['epoch']
        print("last saved model is in epoch {}".format(save_epoch))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs = epochs-save_epoch
    if args.test:
        checkpoint = torch.load(f"./model/saved_model_seed{args.seed}_{args.dataset}")
        save_epoch = checkpoint['epoch']
        print("last saved model is in epoch {}".format(save_epoch))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, candidate, batch_y) in enumerate(tqdm(test_dataloader)):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                candidate = candidate.to(device)
                y_hat = model(batch_x)
                y_hat = y_hat[-1,:,:].gather(1, candidate)

                y_pred += list(y_hat.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        ret = multi_label_classification_task_eval(y_pred, y_true)
        print("result: ", ret)
        with open(f"./result/test_result_seed{args.seed}_{args.dataset}.json", 'w') as f:
            json.dump(ret, f)
        return
    best_aucpr = 0
    for epoch in range(epochs):
        starttime = time.time()
        # 训练
        model.train()
        all_loss = 0.0
        for step, (batch_x, candidate, batch_y) in enumerate(tqdm(train_dataloader)):
            # patients_batch = patients_train[batch_index * batch_size:(batch_index + 1) * batch_size]
            # patients_batch_reshape, patients_lengths = model.padTrainMatrix(patients_batch)  # maxlen × n_samples × inputDimSize
            # batch_x = patients_batch_reshape[0:-1]  # 获取前n-1个作为x，来预测后n-1天的值
            # # batch_y = patients_batch_reshape[1:]
            # batch_y = patients_batch_reshape[1:, :, :283]   # 取出药物作为y
            optimizer.zero_grad()
            # h0 = model.initHidden(batch_x.shape[1])
            # batch_x = torch.stack(batch_x).to(device).float()
            # batch_y = torch.tensor(batch_y).to(device).float()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            candidate = candidate.to(device)
            y_hat = model(batch_x)
            # mask = out_mask2(y_hat, patients_lengths)   # 生成mask,用于将padding的部分输出置0
            # 通过mask，将对应序列长度外的网络输出置0
            # y_hat = y_hat.mul(mask)
            # batch_y = batch_y.mul(mask)
            # # (seq_len, batch_size, out_dim)->(seq_len*batch_size*out_dim, 1)->(seq_len*batch_size*out_dim, )
            # y_hat = y_hat.view(-1,1).squeeze()
            # batch_y = batch_y.view(-1,1).squeeze()
            y_hat = y_hat[-1,:,:].gather(1, candidate)
            loss = loss_mce(y_hat, batch_y)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        print("Train:Epoch-" + str(epoch) + ":" + str(all_loss) + " Train Time:" + str(time.time() - starttime))

        # 测试
        model.eval()
        all_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for step, (batch_x, candidate, batch_y) in enumerate(val_dataloader):
                # patients_batch = patients_test[batch_index * batch_size:(batch_index + 1) * batch_size]
                # patients_batch_reshape, patients_lengths = model.padTrainMatrix(patients_batch)
                # batch_x = patients_batch_reshape[0:-1]
                # batch_y = patients_batch_reshape[1:, :, :283]
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                candidate = candidate.to(device)
                y_hat = model(batch_x)
                y_hat = y_hat[-1,:,:].gather(1, candidate)
                # mask = out_mask2(y_hat, patients_lengths)
                loss = loss_mce(y_hat, batch_y)

                all_loss += loss.item()
                # y_hat = y_hat.detach().cpu().numpy()
                # ndcg, recall, daynum = validation(y_hat, patients_batch, patients_lengths, topk)
                # NDCG += ndcg
                # RECALL += recall
                # DAYNUM += daynum
                # gbert_pred.append(y_hat)
                # gbert_true.append(batch_y.cpu())
                # gbert_len.append(patients_lengths)
                y_pred += list(y_hat.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        print("Test:Epoch-" + str(epoch) + " Loss:" + str(all_loss) + " Test Time:" + str(time.time() - starttime))
        ret = multi_label_classification_task_eval(y_pred, y_true)
        print("eval result: ", ret)
        if ret['auc_pr'] > best_aucpr:
            
            best_auc = ret['auc_pr']
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(state, f"./model/saved_model_seed{args.seed}_{args.dataset}")
            print('\n------------ Save best model ------------\n')
    #test
    checkpoint = torch.load(f"./model/saved_model_seed{args.seed}_{args.dataset}")
    save_epoch = checkpoint['epoch']
    print("last saved model is in epoch {}".format(save_epoch))
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    y_true = []
    y_pred = []
    batch_res_ndcg = []
    batch_res_precision = []
    with torch.no_grad():
        model.eval()
        for step, (batch_x, candidate, batch_y) in enumerate(tqdm(test_dataloader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            candidate = candidate.to(device)
            y_hat = model(batch_x)
            y_hat = y_hat[-1,:,:].gather(1, candidate)
            ndcg = []
            precision = []
            for k in [5,10,20]:
                ndcg.append(ndcg_at_k(y_hat, batch_y, k))
                precision.append(precision_at_k(y_hat,batch_y,k))
            batch_res_ndcg.append(ndcg)
            batch_res_precision.append(precision)
            y_pred += list(y_hat.cpu().detach().numpy().flatten())
            y_true += list(batch_y.cpu().numpy().flatten())
    ndcg_array = np.array(batch_res_ndcg)
    mean_ndcg = np.mean(ndcg_array, axis=0)
    precision_array = np.array(batch_res_precision)
    mean_precision = np.mean(precision_array,axis =0)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    ret = multi_label_classification_task_eval(y_pred, y_true)
    for (k, ndcg, precision) in list(zip([5,10,20], mean_ndcg, mean_precision)):
        ret[f'NDCG@{k}'] = ndcg
        ret[f"Precision@{k}"] = precision
    print("result: ", ret)
    with open(f"./result/test_result_seed{args.seed}_{args.dataset}.json", 'w') as f:
        json.dump(ret, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mimic4')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--resume', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--test', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    train_predict(args)



