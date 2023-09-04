import os
import os.path as osp
import numpy as np
from collections import Counter
import random
import torch
import torch as th
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import Data
from dhgas.utils import move_to, timeit, setup_seed
from dhgas.data.utils import time_select_edge_time as time_select
from dhgas.data.utils import *
from dhgas import config

dataroot = osp.join(config.dataroot, "mag")
processed_datafile = f"{dataroot}/processed"
processed_evaldatafile = f"{dataroot}/processed_eval"


import torch
from dgl.data.utils import load_graphs
import numpy as np
from gensim.models import KeyedVectors


def mp2vec_feat(path, g):
    wordvec = KeyedVectors.load(path, mmap="r")
    for ntype in g.ntypes:
        if ntype == "author":
            prefix = "a_"
        elif ntype == "institution":
            prefix = "i_"
        elif ntype == "field_of_study":
            prefix = "t_"
        else:
            break

        feat = torch.zeros(g.num_nodes(ntype), 128)
        for j in range(g.num_nodes(ntype)):
            try:
                wv = np.array(wordvec[f"{prefix}{j}"])
                feat[j] = torch.from_numpy(wv)
            except KeyError:
                #                 print(f'{prefix}{j}')
                continue

        g.nodes[ntype].data["feat"] = feat

    return g


class MAGDataset:
    def __init__(self, undirected):
        processed = f"{processed_datafile}.pt"
        if osp.exists(processed):
            print(f"loading {processed}")
            dataset = torch.load(processed)
        else:
            dataset = self.preprocess()
            print(f"saving {processed}")
            torch.save(dataset, processed)

        if undirected:
            ToUndirected()(dataset)

        self.dataset = dataset

    def times(self):
        return sorted(
            list(Counter(self.dataset.time_dict["paper"].squeeze().numpy()).keys())
        )

    def preprocess(
        self,
    ):
        glist, label_dict = load_graphs(f"{dataroot}/ogbn_graphs.bin")
        print("loading mp2vec")
        glist = [
            mp2vec_feat(f"{dataroot}/mp2vec/g{i}.vector", g)
            for (i, g) in enumerate(glist)
        ]

        from torch_geometric.data import HeteroData

        data = HeteroData()

        ntypes = glist[0].ntypes
        cetypes = glist[0].canonical_etypes

        def count_ntype_nodes(ntype):
            cnt = np.concatenate([g.ndata["_ID"][ntype].numpy() for g in glist])
            return len(set(list(cnt)))

        def get_feat_dict():
            cnt_types = {ntype: count_ntype_nodes(ntype) for ntype in ntypes}
            feat_dim = glist[0].ndata["feat"][ntypes[0]].shape[1]
            feat_dict = {
                ntype: torch.zeros((cnt_types[ntype], feat_dim)) for ntype in ntypes
            }
            for ntype in ntypes:
                for g_s in glist:
                    node_id = g_s.ndata["_ID"][ntype]
                    node_feat = g_s.ndata["feat"][ntype]
                    feat_dict[ntype][node_id] = node_feat
            return feat_dict

        def get_time():
            ntype = "paper"
            cnt = count_ntype_nodes(ntype)
            time = torch.zeros((cnt, 1)).long()
            for g_s in glist:
                node_id = g_s.ndata["_ID"][ntype]
                node_feat = g_s.ndata["year"]["paper"]
                time[node_id] = node_feat
            return time

        def get_edge_dict():
            edges_dict = {cetype: [] for cetype in cetypes}
            time_dict = {cetype: [] for cetype in cetypes}
            for t, g_s in enumerate(glist):
                for cetype in cetypes:
                    srctype, etype, dsttype = cetype
                    src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
                    edge_index = torch.stack([src, dst])
                    edges_dict[cetype].append(edge_index)
                    time_dict[cetype].append(
                        torch.full((edge_index.shape[1], 1), t).long()
                    )

            for cetype in cetypes:
                edges_dict[cetype] = torch.cat(edges_dict[cetype], dim=1)
                time_dict[cetype] = torch.cat(time_dict[cetype], dim=0)

            return edges_dict, time_dict

        feat_dict = get_feat_dict()
        time = get_time()
        time = time - min(time)
        edges, edge_time = get_edge_dict()

        for ntype, feat in feat_dict.items():
            data[ntype].x = feat
            data[ntype].num_nodes = feat.shape[0]
        data["paper"].time = time
        for etype in edges.keys():
            data[etype].edge_index = edges[etype]
            data[etype].edge_time = edge_time[etype]
        return data


from torch_geometric import seed_everything


@timeit
@torch.no_grad()
def generate_APA(data, device):
    # compress paper
    ei = data["writes"].edge_index.numpy()
    nodes = list(set(ei[1].flatten()))
    n2id = dict(zip(nodes, np.arange(len(nodes))))
    ei[1, :] = np.vectorize(lambda x: n2id[x])(ei[1, :])
    AP_es = torch.LongTensor(ei)

    N1 = data["author"].num_nodes
    N2 = len(nodes)

    AP = (
        torch.sparse_coo_tensor(AP_es, torch.ones(AP_es.shape[1]), (N1, N2))
        .to_dense()
        .to(device)
    )
    PA = AP.t()
    APA = AP @ PA
    APA[torch.eye(APA.shape[0]).bool()] = 0.5
    APA = APA.cpu()
    return APA


@timeit
@torch.no_grad()
def _get_eval_data(glist, idx, device):
    APA_cur = generate_APA(glist[idx], device)
    APA_pre = generate_APA(glist[idx - 1], device)

    APA_pre = (APA_pre > 0.5).float()
    APA_cur = (APA_cur > 0.5).float()

    APA_sub = APA_cur - APA_pre  # new co-author relation
    APA_add = APA_cur + APA_pre
    APA_add[torch.eye(APA_add.shape[0]).bool()] = 0.5

    # get indices of author pairs who collaborate
    indices_true = (APA_sub == 1).nonzero(as_tuple=True)
    indices_false = (APA_add == 0).nonzero(as_tuple=True)

    pos_src = indices_true[0]
    pos_dst = indices_true[1]

    size = int(pos_src.shape[0] * 0.1)

    pos_idx = torch.randperm(pos_src.shape[0])[:size]
    pos_src = pos_src[pos_idx]
    pos_dst = pos_dst[pos_idx]

    neg_src = indices_false[0]
    neg_dst = indices_false[1]

    neg_idx = torch.randperm(neg_src.shape[0])[:size]
    neg_src = neg_src[neg_idx]
    neg_dst = neg_dst[neg_idx]

    pos_edge_index = torch.stack([pos_src, pos_dst])
    neg_edge_index = torch.stack([neg_src, neg_dst])

    # add to edge attr
    data = Data()
    data.edge_label_index = th.cat([pos_edge_index, neg_edge_index], dim=-1).to(device)
    data.edge_label = th.cat(
        [th.ones(pos_edge_index.shape[1]), th.zeros(neg_edge_index.shape[1])], dim=-1
    ).to(device)
    return data


def get_eval_data(glist, device):
    eval_datas = []
    for i in range(len(glist)):
        if i < 1:
            eval_data = []
        else:
            eval_data = _get_eval_data(glist, i, device)
        eval_datas.append(eval_data)
    return eval_datas


class MAGUniDataset:
    def __init__(
        self,
        time_window=1,
        shuffle=True,
        test_full=False,
        is_dynamic=True,
        fix_sample=True,
        all_neg=False,
        seed=22,
        device="cuda:1",
    ):
        self.time_window = time_window
        self.shuffle = shuffle
        self.is_dynamic = is_dynamic
        self.fix_sample = fix_sample
        assert self.fix_sample, "fix_sample must be True"
        assert self.is_dynamic, "is_dynamic must be True"

        setup_seed(seed)  # seed preprocess
        dataset = MAGDataset(undirected=False)

        years = dataset.times()  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset = dataset.dataset  # Hetero
        self.metadata = dataset.metadata()
        self.shift = torch.LongTensor([0])
        self.all_neg = all_neg

        datas = [time_select(dataset, i) for i in years]  # heteros

        processed = f"{processed_evaldatafile}-{seed}.pt"
        if osp.exists(processed):
            print(f"loading {processed}")
            eval_datas = torch.load(processed)
        else:
            print(f"processing {processed}")
            setup_seed(seed)
            eval_datas = get_eval_data(datas, device)
            print(f"saving {processed}")
            torch.save(eval_datas, processed)

        setup_seed(seed)  # seed spliting
        train_idx, val_idx, test_idx = 7, 8, 9

        self.datas = datas
        self.eval_datas = eval_datas
        self.dataset = dataset

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
                time_merge([datas[i] for i in range(val_idx - time_window, val_idx)]),
                eval_datas[val_idx],
            )
        ]
        self.time_dataset["test"] = [
            (
                time_merge([datas[i] for i in range(test_idx - time_window, test_idx)]),
                eval_datas[test_idx],
            )
        ]
        print(
            f"""MAG Dataset(T={len(years)},metadata={self.metadata},dataset={dataset},
            )"""
        )

    @property
    def test_dataset(self):
        return self.time_dataset["test"][0]

    @property
    def val_dataset(self):
        return self.time_dataset["val"][0]

    @property
    def train_dataset(self):
        data = [d[0] for d in self.time_dataset["train"]]
        eval_datas = [d[1] for d in self.time_dataset["train"]]
        ret = list(zip(data, eval_datas))
        if self.shuffle:
            random.shuffle(ret)
        return ret

    def to(self, device):
        self.time_dataset = move_to(self.time_dataset, device)


if __name__ == "__main__":
    device = "cuda:1"
    # mag=MAGDataset(undirected=False)
    # edge_types=mag.dataset.edge_types
    # mag.dataset[edge_types[0]]
    mag = MAGUniDataset(time_window=3, device=device, seed=11)
    mag.dataset.edges

    # dataset=mag.dataset
    # years=mag.times()
    # datas = [time_select(dataset, i) for i in years]  # heteros
    # data=datas[0]
    # data

    # link_data=get_eval_data(datas,device)
