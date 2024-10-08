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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score
import argparse
import json
from mydataloader import EvalDataset
import torch.utils.data as data_utils
import random
from pathlib import Path
from tqdm import tqdm
def ndcg_at_k(pred, true, k):
    # 排序每个 batch 的预测值，并返回 top-k 的 indices
    _, indices = torch.topk(pred, k=k, dim=1)
    
    # 获取真实值中 top-k 的 relevant 值
    batch_size = true.size(0)
    item_num = true.size(1)
    
    # 创建一个mask来表示top-k中的真实相关值
    relevant_mask = torch.gather(true, dim=1, index=indices)
    
    # 计算 DCG: Discounted Cumulative Gain
    discount = 1 / torch.log2(torch.tensor([2 for _ in range(k)]).float().to(pred.device))  # 从2开始是因为log2(1)是不定义的
    dcg = (relevant_mask * discount).sum(dim=1)
    
    # 计算理想的 DCG: IDCG
    ideal_relevant_mask = torch.sort(true, dim=1, descending=True)[0][:, :k]
    idcg = (ideal_relevant_mask * discount).sum(dim=1)
    
    ideal_relevant_mask = torch.sort(true, dim=1, descending=True)[0][:, :k]  # 理想情况是 top-k 的 relevant 值排序
    idcg = (ideal_relevant_mask * discount).sum(dim=1)

    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0  # 处理分母为0的情况
    return ndcg.mean().item()  # 返回 batch 的平均 NEDG@k

def precision_at_k(pred, true, k):
    # 排序每个 batch 的预测值，并返回 top-k 的 indices
    _, indices = torch.topk(pred, k=k, dim=1)
    
    # 获取真实值中 top-k 的 relevant 值
    relevant_mask = torch.gather(true, dim=1, index=indices)
    
    # 计算 Precision@k: top-k 中 relevant 项的数量 / k
    precision = relevant_mask.sum(dim=1).float() / k
    
    return precision.mean().item()  # 返回 batch 的平均 Precision@k

def multi_label_classification_task_eval(pred, targets, threshold=0.5):
    pred_binary = (pred > 0.5).astype(int)

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

def test(args):
    batch_size = 100
    device = torch.device("cuda:7" if torch.cuda.is_available() == True else 'cpu')
    data = load_data(args.dataset)
    test_data = data['test']
    smap_l = data['smap_l']

    test_dataset = EvalDataset(test_data, smap_l)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = Dipole(input_dim=1, day_dim=100, rnn_hiddendim=300, output_dim=len(data['smap_l'])+1)
    model.to(device)
    checkpoint = torch.load(f"./model/saved_model_seed{args.seed}_{args.dataset}")
    save_epoch = checkpoint['epoch']
    print("last saved model is in epoch {}".format(save_epoch))
    model.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
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
    args = parser.parse_args()
    test(args)