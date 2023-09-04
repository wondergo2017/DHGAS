from dhgas.data.utils import time_select_edge_time as time_select
import torch_geometric.transforms as T
from dhgas.utils import move_to, setup_seed
import random
import numpy as np
from dhgas.data.utils import *
from torch_geometric.data import HeteroData, Data
import torch as th
from torch import Tensor, LongTensor, FloatTensor
from torch_geometric.transforms import ToUndirected
from dhgas.data.utils import time_merge_edge_time
from dhgas import config
import os.path as osp

dataroot = osp.join(config.dataroot, "ecomm")
datafile = osp.join(dataroot, "ecomm_edge_train.txt")


class EcommDataset:
    def __init__(self, undirected=True):
        # 1. read data
        data = []
        with open(datafile, "r") as file:
            for line in file:
                nid1, nid2, etype, t = line.split()
                data.append([nid1, nid2, etype, int(t)])
        data = np.array(data)
        # 2. reorder index
        # WARNING : here shoud ensure that time order is right.
        data = setorderidx(
            data
        )  # this function here is right because the time from 20190610 to 20190619. same for str order.
        # 3. statistics
        cnt = {}
        for i in range(4):
            cnt[i] = Counter(data[:, i])
        self.years = list(cnt[3].keys())
        node1_num = len(cnt[0])
        node2_num = len(cnt[1])
        # four types
        etypes = [
            tuple("user click item".split()),
            tuple("user buy item".split()),
            tuple("user cart item".split()),
            tuple("user favorite item".split()),
            tuple("user interact item".split()),
        ]
        # 4 put in HeteroData
        dataset = HeteroData()
        for i in range(4):
            dataset[etypes[i]].edge_index = LongTensor(data[data[:, 2] == i][:, :2].T)
            dataset[etypes[i]].edge_time = LongTensor(data[data[:, 2] == i][:, [3]])
        dataset[etypes[-1]].edge_index = torch.cat(
            [dataset[etypes[i]].edge_index for i in range(4)], dim=1
        )
        dataset[etypes[-1]].edge_time = torch.cat(
            [dataset[etypes[i]].edge_time for i in range(4)]
        )

        dataset["user"].x = th.arange(0, node1_num).unsqueeze(-1)
        dataset["item"].x = th.arange(0, node2_num).unsqueeze(-1)

        if undirected:
            dataset = ToUndirected()(dataset)

        self.dataset = dataset

    def times(self):
        return self.years


def hetero_remove_edges_unseen_nodes(data, etype, train_nodes0, train_nodes1):
    """inplace operation, remove edges with nodes not in train_nodes"""
    idxs = []
    ei = data[etype].edge_index.T  # [E,2]
    # print(f'before removing : {ei.T.shape}')
    for i in range(ei.shape[0]):
        e = ei[i].numpy()
        if (e[0] in train_nodes0) and (e[1] in train_nodes1):
            idxs.append(i)
    idxs = torch.LongTensor(idxs)
    data[etype].edge_index = torch.index_select(data[etype].edge_index, 1, idxs)
    # print(f'after removing : {data[etype].edge_index.shape}')


def get_eval_data(data):
    eval_data = HeteroData()
    # for x in ['user','item']:
    #     eval_data[x]=data[x]
    eval_data[tuple("user interact item".split())].edge_index = data[
        "interact"
    ].edge_index
    return eval_data


class EcommUniDataset:
    def __init__(
        self, time_window=1, shuffle=True, test_full=False, is_dynamic=True, seed=22
    ):
        self.time_window = time_window
        self.shuffle = shuffle
        self.is_dynamic = is_dynamic

        setup_seed(seed)  # seed preprocess
        dataset = EcommDataset(undirected=True)

        setup_seed(seed)  # seed spliting
        years = dataset.times()  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset = dataset.dataset  # Hetero

        # id_embdding:
        for nt in "user item".split():
            dataset[nt].x = dataset[nt].x.squeeze(-1)

        datas = [time_select(dataset, i) for i in years]  # heteros
        eval_datas = [get_eval_data(data) for data in datas]  # homo

        self.eval_etype = eval_etype = tuple("user interact item".split())
        # delete type interact
        for i in years:
            del datas[i]["interact"], datas[i]["rev_interact"]
        del dataset["interact"], dataset["rev_interact"]

        train_idx, val_idx, test_idx = 7, 8, 9

        # remove edges of nodes unseen in training
        train_nodes0 = [set()]  # [null,edge0,edge1,...]
        train_nodes1 = [set()]  # [null,edge0,edge1,...]
        for i in range(train_idx):
            train_i0 = train_nodes0[-1] | set(
                eval_datas[i][eval_etype].edge_index[0].unique().numpy()
            )
            train_i1 = train_nodes1[-1] | set(
                eval_datas[i][eval_etype].edge_index[1].unique().numpy()
            )
            train_nodes0.append(train_i0)
            train_nodes1.append(train_i1)

        for i in range(1, train_idx):
            hetero_remove_edges_unseen_nodes(
                eval_datas[i], eval_etype, train_nodes0[i], train_nodes1[i]
            )
        for i in range(train_idx, test_idx + 1):
            hetero_remove_edges_unseen_nodes(
                eval_datas[i], eval_etype, train_nodes0[-1], train_nodes1[-1]
            )

        # negative sampling
        eval_datas = [
            hetero_linksplit(eval_datas[k], eval_etype) for k in years
        ]  # homo

        self.datas = datas
        self.eval_datas = eval_datas
        self.dataset = dataset
        self.metadata = dataset.metadata()

        if is_dynamic:
            time_merge = lambda x: x
        else:
            time_merge = time_merge_edge_time

        self.time_dataset = {}
        self.time_dataset["train"] = [
            (time_merge([datas[i] for i in range(k - time_window, k)]), eval_datas[k])
            for k in range(time_window, train_idx + 1)
        ]
        self.time_dataset["val"] = [
            (
                time_merge([datas[i] for i in range(val_idx)])
                if test_full
                else time_merge(
                    [datas[i] for i in range(val_idx - time_window, val_idx)]
                ),
                eval_datas[val_idx],
            )
        ]
        self.time_dataset["test"] = [
            (
                time_merge([datas[i] for i in range(test_idx)])
                if test_full
                else time_merge(
                    [datas[i] for i in range(test_idx - time_window, test_idx)]
                ),
                eval_datas[test_idx],
            )
        ]

        print(
            f"""Ecomm Dataset(T={len(years)},metadata={self.metadata},dataset={dataset},
            )"""
        )

    @property
    def test_dataset(self):
        test_data = self.time_dataset["test"][0][0]
        test_eval = self.time_dataset["test"][0][1]["interact"]
        return test_data, test_eval

    @property
    def val_dataset(self):
        val_data = self.time_dataset["val"][0][0]
        val_eval = self.time_dataset["val"][0][1]["interact"]
        return val_data, val_eval

    @property
    def train_dataset(self):
        data = [d[0] for d in self.time_dataset["train"]]
        [self.shift_negetive_sample(d[1]) for d in self.time_dataset["train"]]
        eval_datas = [d[1]["interact"] for d in self.time_dataset["train"]]
        ret = list(zip(data, eval_datas))
        if self.shuffle:
            random.shuffle(ret)
        return ret

    def shift_negetive_sample(self, coauthor_data):
        hetero_linksplit(coauthor_data, self.eval_etype, inplace=True)

    def to(self, device):
        self.time_dataset = move_to(self.time_dataset, device)


if __name__ == "__main__":
    time_window = 5
    shuffle = True
    test_full = False
    is_dynamic = False
    dataset = EcommUniDataset(
        time_window=time_window,
        shuffle=shuffle,
        test_full=test_full,
        is_dynamic=is_dynamic,
    )

    dataset.train_dataset
    dataset.dataset
    dataset.metadata
    dataset.eval_datas
    dataset.train_dataset
    dataset.val_dataset[1]
    dataset.train_dataset[0][1]

    dataset.dataset
