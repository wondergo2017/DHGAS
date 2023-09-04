import os
import os.path as osp
import numpy as np
from collections import Counter
import random
import torch
import torch as th
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from dhgas.utils import move_to, timeit, setup_seed
from dhgas.data.metapath2vec import get_metapath2vec
from dhgas.data.utils import time_select_edge_time as time_select
from dhgas.data.utils import *
from dhgas import config

fnames = ["Database", "Data Mining", "Medical Informatics", "Theory", "Visualization"]
dataroot = osp.join(config.dataroot, "Cross-Domain_data")
datafiles = [f"{dataroot}/{name}.txt" for name in fnames]
word2vec_file = f"{dataroot}/abs2vec"
processed_datafile = f"{dataroot}/processed"


def parse(datafile):
    field = os.path.split(datafile)[-1]
    with open(datafile, "r") as file:
        lines = file.readlines()
    lines[0].split("\t")
    papers = []
    for line in lines:
        (venue, title, authors, year, abstract) = line.split("\t")
        try:
            year = int(year)
            paper = (venue, title, authors, year, abstract, field)
        except Exception as e:
            print(e)
        papers.append(paper)
    papers = np.array(papers)
    return papers


class CrossDomainDataset:
    """Aminer CrossDomain Dataset
    we use gensim.word2vec
    """

    def __init__(
        self, undirected=True, metapath2vec=False, word2vec_size=32, device="auto"
    ):
        self.device = device
        self.metapath2vec = metapath2vec
        processed = f"{processed_datafile}-{metapath2vec}-{word2vec_size}.pt"
        if osp.exists(processed):
            # print(f'loading {processed}')
            dataset = torch.load(processed)
        else:
            dataset = self.preprocess(word2vec_size)
            # print(f'saving {processed}')
            torch.save(dataset, processed)

        if undirected:
            ToUndirected()(dataset)

        self.dataset = dataset

    def times(self):
        return sorted(
            list(Counter(self.dataset.time_dict["paper"].squeeze().numpy()).keys())
        )

    def clear_file(self):
        pass

    def _metapath2vec(self, data, emb_size=32):
        metapath = [
            ("author", "rev_written", "paper"),
            ("paper", "published", "venue"),
            ("venue", "rev_published", "paper"),
            ("paper", "written", "author"),
        ]
        device = self.device
        data = ToUndirected()(data.clone())
        nodevecs = get_metapath2vec(
            data, metapath, epochs=1, emb_size=emb_size, device=device
        )
        return nodevecs

    def preprocess(self, word2vec_size=32):
        data = self.preprocess1(word2vec_size=32)
        if self.metapath2vec:
            nodevecs = self._metapath2vec(data, emb_size=word2vec_size)
            data["author"].x = nodevecs["author"]
            data["venue"].x = nodevecs["venue"]
        return data

    def preprocess1(self, word2vec_size=32):
        papers = []
        for file in datafiles:
            paper = parse(file)
            papers.append(paper)

        for i, paper in enumerate(papers):
            print(fnames[i])
            print(Counter(paper[:, 3]))
        papers = np.concatenate(papers)

        Counter(papers[:, 3])  # year
        Counter(papers[:, 0])  # venue

        authors = []
        for paper in papers:
            authors.extend(paper[2].split(","))
        len(authors)  # authors

        # do mapping
        vid2vname = list(Counter(papers[:, 0]).keys())
        vname2vid = map2id(vid2vname)

        authors = []
        for paper in papers:
            authors.extend(paper[2].split(","))
        aid2aname = list(sorteddict(Counter(authors), min=False).keys())
        aname2aid = map2id(aid2aname)
        aname2aid

        yid2yname = sorted(list(map(int, Counter(papers[:, 3]).keys())))
        yname2yid = map2id(yid2yname)
        print("tid2tname:", yname2yid)

        fid2fname = sorted(list(Counter(papers[:, 5]).keys()))
        fname2fid = map2id(fid2fname)

        # venue link
        e_pv = []
        for i, vname in enumerate(papers[:, 0]):
            e_pv.append([i, vname2vid[vname]])
        e_pv = th.LongTensor(np.array(e_pv)).T

        # author link
        e_pa = []
        for i, anames in enumerate(papers[:, 2]):
            for aname in anames.split(","):
                e_pa.append([i, aname2aid[aname]])
        e_pa = th.LongTensor(np.array(e_pa)).T

        # title; we do not use
        x_title = papers[:, 1]

        # years
        x_year = th.LongTensor(list(map(lambda x: yname2yid[int(x)], papers[:, 3])))
        x_year

        # field
        x_field = th.LongTensor(list(map(lambda x: fname2fid[x], papers[:, 5])))
        x_field

        # abstract
        x_abstract = papers[:, 4]
        emb_file = f"{word2vec_file}-{word2vec_size}.npy"
        if os.path.exists(emb_file):
            print(f"loading {emb_file}")
            emb_abs = np.load(emb_file)
        else:
            print(f"generating {emb_file}")
            emb_abs = sen2vec(x_abstract, vector_size=word2vec_size)
            np.save(emb_file, emb_abs)

        # author
        num_author = len(set(e_pa[1, :].numpy()))
        x_author = torch.arange(num_author)
        x_author

        # venue
        num_venue = len(set(e_pv[1, :].numpy()))
        x_venue = torch.arange(num_venue)
        x_venue

        data = HeteroData()
        data["paper", "published", "venue"].edge_index = e_pv
        data["paper", "written", "author"].edge_index = e_pa
        data["paper"].x = torch.FloatTensor(emb_abs)
        data["paper"].y = x_field
        data["paper"].time = x_year.unsqueeze(-1)
        data["author"].x = x_author.unsqueeze(-1)
        data["venue"].x = x_venue.unsqueeze(-1)
        data["paper"].num_nodes = data["published"].edge_index.shape[1]

        data["published"].edge_time = data["paper"].time.index_select(0, e_pv[0, :])
        data["written"].edge_time = data["paper"].time.index_select(0, e_pa[0, :])

        info_dict = {
            "vid2vname": vid2vname,
            "vname2vid": vname2vid,
            "aid2aname": aid2aname,
            "aname2aid": aname2aid,
            "yid2yname": yid2yname,
            "yname2yid": yname2yid,
        }
        return data


from dhgas.data.utils import time_merge_edge_time
from dhgas.data.utils import negative_sample


def remove_edges_unseen_nodes(data, train_nodes):
    """inplace operation, remove edges with nodes not in train_nodes"""
    idxs = []
    ei = data.edge_index.T  # [E,2]
    # print(f'before removing : {ei.T.shape}')
    for i in range(ei.shape[0]):
        e = ei[i].numpy()
        if (e[0] in train_nodes) and (e[1] in train_nodes):
            idxs.append(i)
    idxs = torch.LongTensor(idxs)
    data.edge_index = torch.index_select(data.edge_index, 1, idxs)
    # print(f'after removing : {data.edge_index.shape}')


class CrossDomainUniDataset:
    def __init__(
        self,
        time_window=1,
        shuffle=True,
        test_full=False,
        is_dynamic=True,
        fix_sample=False,
        all_neg=False,
        seed=22,
    ):
        self.time_window = time_window
        self.shuffle = shuffle
        self.is_dynamic = is_dynamic
        self.fix_sample = fix_sample

        setup_seed(seed)  # seed preprocess
        dataset = CrossDomainDataset(undirected=True)

        setup_seed(seed)  # seed spliting
        years = (
            dataset.times()
        )  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        dataset = dataset.dataset  # Hetero
        self.metadata = dataset.metadata()
        self.shift = torch.LongTensor([0])
        self.all_neg = all_neg

        # id embedding
        for nt in "author venue".split():
            dataset[nt].x = dataset[nt].x.squeeze(-1)

        datas = [time_select(dataset, i) for i in years]  # heteros
        eval_datas = [get_author_graph(datas[k]) for k in years]  # localidxs

        train_idx, val_idx, test_idx = 13, 14, 15

        # remove edges with nodes not in train_nodes
        train_nodes = [set()]  # [null,edge0,edge1,...]
        for i in range(train_idx):  # [0-12]
            train_i = train_nodes[-1] | set(eval_datas[i].edge_index.unique().numpy())
            train_nodes.append(train_i)

        for i in range(1, train_idx):
            remove_edges_unseen_nodes(
                eval_datas[i], train_nodes[i]
            )  # remove for [1,12]
        for i in range(train_idx, test_idx + 1):
            remove_edges_unseen_nodes(
                eval_datas[i], train_nodes[-1]
            )  # remove for [13-15]

        # negative sampling
        eval_datas = [
            linksplit(eval_datas[k], self.all_neg) for k in years
        ]  # get neg edges

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
            for k in range(time_window, train_idx + 1)  # [0-12] -> [13]
        ]

        collect_val = time_merge_edge_time(
            [datas[i] for i in range(val_idx - time_window + 1)]
        )
        collect_test = time_merge_edge_time(
            [datas[i] for i in range(test_idx - time_window + 1)]
        )

        self.time_dataset["val"] = [
            (
                time_merge([datas[i] for i in range(val_idx)])
                if test_full
                else time_merge(
                    [collect_val]
                    + [datas[i] for i in range(val_idx - time_window + 1, val_idx)]
                ),
                eval_datas[val_idx],
            )
        ]
        self.time_dataset["test"] = [
            (
                time_merge([datas[i] for i in range(test_idx)])
                if test_full
                else time_merge(
                    [collect_test]
                    + [datas[i] for i in range(test_idx - time_window + 1, test_idx)]
                ),
                eval_datas[test_idx],
            )
        ]
        print(
            f"""CrossDomain Dataset(T={len(years)},metadata={self.metadata},dataset={dataset},
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
        if not self.fix_sample:
            [self.shift_negetive_sample(d[1]) for d in self.time_dataset["train"]]
        eval_datas = [d[1] for d in self.time_dataset["train"]]
        ret = list(zip(data, eval_datas))
        if self.shuffle:
            random.shuffle(ret)
        return ret

    # @timeit
    def shift_negetive_sample(self, coauthor_data):
        # negative_sample(coauthor_data)
        linksplit(coauthor_data, inplace=True)

    def to(self, device):
        self.time_dataset = move_to(self.time_dataset, device)


def train_val_test_split(maxn, val_ratio=0.1, test_ratio=0.1):
    val_num = int(np.ceil(val_ratio * maxn))
    test_num = int(np.ceil(test_ratio * maxn))
    train_num = maxn - val_num - test_num
    assert train_num >= 0 and test_num >= 0 and val_num >= 0

    idxs = np.arange(maxn)
    np.random.shuffle(idxs)
    test_idxs = idxs[:test_num]
    val_idxs = idxs[test_num : test_num + val_num]
    train_idxs = idxs[test_num + val_num :]
    print(f"split sizes: train {train_num} ; val {val_num} ; test {test_num}")

    train_mask = torch.zeros(maxn).bool()
    train_mask[train_idxs] = True
    val_mask = torch.zeros(maxn).bool()
    val_mask[val_idxs] = True
    test_mask = torch.zeros(maxn).bool()
    test_mask[test_idxs] = True

    return train_mask, val_mask, test_mask


def get_eval_data(data, times):
    dtime = data["paper"].time.flatten()
    mask = dtime.new_zeros((dtime.shape[0], 1)).flatten().bool()
    for t in times:
        mask = mask | (dtime == t)
    eval_data = Data()
    eval_data.y = data["paper"].y
    eval_data.mask = mask
    return eval_data


if __name__ == "__main__":
    time_window = 5
    shuffle = True
    test_full = False
    is_dynamic = False
    dataset = CrossDomainUniDataset(
        time_window=time_window,
        shuffle=shuffle,
        test_full=test_full,
        is_dynamic=is_dynamic,
    )
    data = dataset.datas
    data[0]
    dataset.dataset
    dataset.train_dataset[0]

    dataset.dataset["author"].x

    dataset.datas[6].edge_time_dict
    dataset.val_dataset[0].edge_time_dict
