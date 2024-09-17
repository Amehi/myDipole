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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

def binary_classification_task_eval(logits, targets, threshold=0.5):
    # 将 logits 转为概率
    probs = torch.sigmoid(logits)

    # 二值化预测
    preds = (probs > threshold).float()

    # 转为 numpy 数组
    preds_np = preds.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    probs_np = probs.cpu().numpy().flatten()

    # 计算 precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(targets_np, preds_np, average='binary')

    # 计算 AUC-PR 和 AUC-ROC
    auc_pr = average_precision_score(targets_np, probs_np)
    auc_roc = roc_auc_score(targets_np, probs_np)

    return precision, recall, f1, auc_pr, auc_roc

class Dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self.name = name

    def __getitem__(self, index):#返回的是tensor
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def recall_at_k(output, target, k):
    batch_size = output.size(0)
    recall_sum = 0.0

    for i in range(batch_size):
        # 对每个样本的输出进行排序，并取前 k 个预测物品的索引
        _, topk_indices = torch.topk(output[i], k)
        
        # 获取这些索引在 ground truth 中的值（0 或 1）
        topk_ground_truth = target[i][topk_indices]
        
        # 计算在前 k 个预测物品中正确预测的数量
        num_relevant = topk_ground_truth.sum().item()
        
        # 计算 recall
        num_relevant_total = target[i].sum().item()
        recall = num_relevant / num_relevant_total if num_relevant_total != 0 else 0.0
        
        recall_sum += recall

    return recall_sum / batch_size


def test(batch_size=100, seed=2024):
    device = torch.device("cuda:1" if torch.cuda.is_available() == True else 'cpu')
    with open("out/Dipole/test.pkl", 'rb') as f:
        test_raw = pickle.load(f)
    test_dataset = Dataset(test_raw[0], test_raw[1])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    model = Dipole(input_dim=1312, day_dim=200, rnn_hiddendim=300, output_dim=1312)
    model = model.cuda(device=1)
    model.load_state_dict(torch.load(f"./model/saved_model_seed{seed}")['net'])
    model.eval()
    with torch.no_grad():
        batch_res = []
        for step, (batch_x, batch_y) in enumerate(test_loader):
            # patients_batch = patients_test[batch_index * batch_size:(batch_index + 1) * batch_size]
            # patients_batch_reshape, patients_lengths = model.padTrainMatrix(patients_batch)
            # batch_x = patients_batch_reshape[0:-1]
            # batch_y = patients_batch_reshape[1:, :, :283]
            batch_x = torch.stack(batch_x).to(device).float()
            batch_y = torch.tensor(batch_y).to(device).float()
            y_hat = model(batch_x)

            precision, recall, f1, auc_pr, auc_roc = binary_classification_task_eval(y_hat[-1,:,:], batch_y)
            batch_res.append([precision, recall, f1, auc_pr, auc_roc])
    cur_recall = np.array(batch_res).mean(axis=0)
    print("result: ", cur_recall)

    with open(f'./model/test_res_seed{seed}.txt', 'w') as f:
        f.write(np.array2string(cur_recall, precision=6, separator=',',
                suppress_small=True))

if __name__ == "__main__":
    test(seed=2026)
    test(seed=2027)
    test(seed=2028)