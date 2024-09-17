import torch
import torch.utils.data as data_utils

class TrainDataset(data_utils.Dataset):
    def __init__(self, train, smap_l):
        self.trajs = train[0]
        self.gds = train[1]
        self.max_len = 100
        self.smap_l = smap_l
    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.trajs[index]
        gd = self.gds[index]
        candidates = list(self.smap_l.values())
        labels = [0] * len(self.smap_l)
        for idx in range(len(candidates)):
            if candidates[idx] in gd:
                labels[idx] = 1

        
        traj = traj[-self.max_len:]

        mask_len = self.max_len - len(traj)

        traj = [0] * mask_len + traj

        traj = list(map(lambda x: [x], traj))
        return torch.LongTensor(traj), torch.LongTensor(candidates), torch.LongTensor(labels)
    
class EvalDataset(data_utils.Dataset):
    def __init__(self, eval,smap_l):
        self.trajs = eval[0]
        self.gds = eval[1]
        self.max_len = 100
        self.smap_l = smap_l

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        candidates = list(self.smap_l.values())

        traj = self.trajs[index]
        gd = self.gds[index]

        labels = [0] * len(self.smap_l)
        for idx in range(len(candidates)):
            if candidates[idx] in gd:
                labels[idx] = 1
        
        traj = traj[-self.max_len:]
        # gd = gd[-100:]
        mask_len_traj = self.max_len - len(traj)
        # mask_len_gd = 100 - len(gd)
        traj = [0] * mask_len_traj + traj
        # gd = [0] * mask_len_gd + gd
        traj = list(map(lambda x: [x], traj))
        return torch.LongTensor(traj), torch.LongTensor(candidates), torch.LongTensor(labels)