import os
import os.path as osp
import numpy as np
from collections import Counter
import torch
import torch as th
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from dhgas.utils import move_to, timeit, setup_seed
from dhgas.data.metapath2vec import get_metapath2vec
from dhgas.data.utils import time_select_edge_time as time_select
from dhgas.data.utils import *
import json
from collections import Counter
from dhgas import config

dataroot = "./raw_data/"
processed_datafile = osp.join(config.dataroot, f"yelp/processed/")

datafiles = {}
for part in "business review tip user checkin".split():
    datafiles[part] = os.path.join(dataroot, f"yelp_academic_dataset_{part}.json")


def get_raw_data(file):
    rawdata = {}
    data = []
    with open(file["business"], "r") as f:
        for line in f:
            d = json.loads(line)
            data.append([d["business_id"], d["categories"]])
    rawdata["business"] = data  # [bid,cate]

    data = []
    with open(file["review"], "r") as f:
        for line in f:
            d = json.loads(line)
            data.append(
                [
                    d["user_id"],
                    d["business_id"],
                    d["stars"],
                    d["date"],
                    # d['text']
                ]
            )
    rawdata["review"] = data  # [uid,bid,rate,date]

    data = []
    with open(file["tip"], "r") as f:
        for line in f:
            d = json.loads(line)
            data.append(
                [
                    d["user_id"],
                    d["business_id"],
                    d["date"],
                    # d['text']
                ]
            )
    rawdata["tip"] = data  # [uid,bid,date]
    print(
        f'loading Yelp raw data : #business {len(rawdata["business"])} #review {len(rawdata["review"])} #tips {len(rawdata["tip"])}'
    )
    return rawdata


def select_business(data, cates_included):
    cate = [x[1] for x in data]
    cnt = Counter(cate)
    cnt = dict(sorted(cnt.items(), key=lambda item: -item[1]))
    business_included = []
    for x in data:
        try:
            bid, cat = x
            for cat_in in cates_included:
                if cat and cat_in in cat:
                    business_included.append([bid, cat_in])
                    break
        except Exception as e:
            print(x)
            print(e)
    bid_cnt = Counter([x[1] for x in business_included])
    bid_set = set(x[0] for x in business_included)
    print(f"select business in {cates_included}")
    print(f"#business {len(bid_set)}")
    business_included = np.array(business_included)
    return business_included, bid_set


def parse_date(date, year=2012):
    y, m = date[:7].split("-")
    y = int(y)
    m = int(m)
    if y == year:
        return m
    return None


def select_reviews(data, bid_set):
    reviews = []
    for x in data:
        uid, bid, rate, date = x
        date = parse_date(date)
        if date and bid in bid_set:
            review = [uid, bid, date]
            reviews.append(review)
    print(f"#reviews {len(reviews)}")
    reviews = np.array(reviews)
    return reviews


def select_tips(data, bid_set):
    tips = []
    for x in data:
        uid, bid, date = x
        date = parse_date(date)
        if date and bid in bid_set:
            tip = [uid, bid, date]
            tips.append(tip)
    print(f"#tips {len(tips)}")
    tips = np.array(tips)
    return tips


class YelpDataset:
    def __init__(
        self, undirected=True, metapath2vec=False, word2vec_size=32, device="auto"
    ):
        self.device = device
        self.metapath2vec = metapath2vec
        processed = osp.join(processed_datafile, f"{metapath2vec}-{word2vec_size}.pt")
        if osp.exists(processed):
            # print(f'loading {processed}')
            dataset = torch.load(processed)
        else:
            os.makedirs(processed_datafile, exist_ok=True)
            dataset = self.preprocess(word2vec_size)
            # print(f'saving {processed}')
            torch.save(dataset, processed)

        if undirected:
            ToUndirected()(dataset)

        self.dataset = dataset

    def times(self):
        return sorted(
            list(Counter(self.dataset["review"].edge_time.squeeze().numpy()).keys())
        )

    def clear_file(self):
        pass

    def _metapath2vec(self, data, emb_size=32):
        metapath = [
            ("user", "review", "item"),
            ("item", "rev_review", "user"),
            ("user", "tip", "item"),
            ("item", "rev_tip", "user"),
        ]
        device = self.device
        data = ToUndirected()(data.clone())
        nodevecs = get_metapath2vec(
            data, metapath, epochs=1, emb_size=emb_size, device=device
        )
        return nodevecs

    def preprocess(self, word2vec_size=32):
        data = self.preprocess1()
        if self.metapath2vec:
            nodevecs = self._metapath2vec(data, emb_size=word2vec_size)
            data["user"].x = nodevecs["user"]
            data["item"].x = nodevecs["item"]
        return data

    def preprocess1(self):
        rawdata = get_raw_data(datafiles)
        cates_included = [
            "American (New)",
            "Fast Food",
            "Sushi Bars",
            # "Coffee & Tea",
            # "Pizza",
            # "Hair Salons",
            # "Automotive",
            # "Banks & Credit Unions",
            # "Apartments",
            # "Barbers",
        ]
        business, bid_set = select_business(
            rawdata["business"], cates_included
        )  # [bid,cate]
        reviews = select_reviews(rawdata["review"], bid_set)  # [uid,bid,date]
        tips = select_tips(rawdata["tip"], bid_set)  # [uid,bid,date]

        # do mapping
        bid2bname = list(Counter(business[:, 0]).keys())
        bname2bid = map2id(bid2bname)
        bname2cname = dict(x for x in business)

        cid2cname = list(Counter(business[:, 1]).keys())
        cname2cid = map2id(cid2cname)

        users = list(reviews[:, 0]) + list(tips[:, 0])
        uid2uname = list(Counter(users).keys())
        uname2uid = map2id(uid2uname)

        times = list(reviews[:, 2]) + list(tips[:, 2])
        tid2tname = sorted(
            list(map(int, list(Counter(times).keys())))
        )  # make sure times order!
        tname2tid = map2id(tid2tname)
        print("tname2tid:", tname2tid)

        # review link
        e_review = []
        e_review_time = []
        for i, (u, b, t) in enumerate(reviews):
            e_review.append([uname2uid[u], bname2bid[b]])
            e_review_time.append(tname2tid[int(t)])
        e_review = th.LongTensor(np.array(e_review)).T
        e_review_time = th.LongTensor(e_review_time)

        # tip link
        e_tip = []
        e_tip_time = []
        for i, (u, b, t) in enumerate(tips):
            e_tip.append([uname2uid[u], bname2bid[b]])
            e_tip_time.append(tname2tid[int(t)])
        e_tip = th.LongTensor(np.array(e_tip)).T
        e_tip_time = th.LongTensor(e_tip_time)

        # put into heterodata
        # edge
        etypes = [
            tuple("user review item".split()),
            tuple("user tip item".split()),
            tuple("user interact item".split()),
        ]
        dataset = HeteroData()
        dataset[tuple("user review item".split())].edge_index = e_review
        dataset[tuple("user review item".split())].edge_time = e_review_time
        dataset[tuple("user tip item".split())].edge_index = e_tip
        dataset[tuple("user tip item".split())].edge_time = e_tip_time

        dataset[etypes[-1]].edge_index = torch.cat(
            [dataset[etypes[i]].edge_index for i in range(2)], dim=1
        )
        dataset[etypes[-1]].edge_time = torch.cat(
            [dataset[etypes[i]].edge_time for i in range(2)]
        )

        # node feature
        node1_num = len(uid2uname)
        node2_num = len(bid2bname)
        dataset["user"].x = th.arange(0, node1_num).unsqueeze(-1)
        dataset["item"].x = th.arange(0, node2_num).unsqueeze(-1)
        dataset["item"].y = th.LongTensor(
            [cname2cid[bname2cname[bid2bname[bid]]] for bid in range(len(bid2bname))]
        )

        return dataset


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


class YelpNCLFDataset:
    def __init__(
        self,
        time_window=1,
        shuffle=True,
        test_full=False,
        is_dynamic=True,
        seed=22,
        val_ratio=0.1,
        test_ratio=0.1,
    ):
        self.time_window = time_window
        self.shuffle = shuffle
        self.is_dynamic = is_dynamic
        self.test_full = test_full

        setup_seed(seed)  # seed preprocess
        dataset = YelpDataset(undirected=True, metapath2vec=True)

        setup_seed(seed)  # seed spliting
        times = dataset.times()  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        dataset = dataset.dataset  # Hetero
        self.metadata = dataset.metadata()

        datas = [time_select(dataset, i) for i in times]  # heteros

        def get_eval_data(dataset, mask):
            eval_data = Data()
            eval_data.y = dataset["item"].y
            eval_data.mask = mask
            return eval_data

        train_mask, val_mask, test_mask = train_val_test_split(
            len(dataset["item"].x), val_ratio=val_ratio, test_ratio=test_ratio
        )

        train_eval = get_eval_data(dataset, train_mask)
        val_eval = get_eval_data(dataset, val_mask)
        test_eval = get_eval_data(dataset, test_mask)

        self.eval_datas = [train_eval, val_eval, test_eval]
        self.datas = datas
        self.dataset = dataset

        if is_dynamic:
            time_merge = lambda x: x
        else:
            time_merge = time_merge_edge_time

        maxn = 12
        patchlen = maxn // time_window
        self.time_dataset = time_merge(
            [
                time_merge_edge_time([datas[i] for i in range(k, k + patchlen)])
                for k in range(0, maxn, patchlen)
            ]
        )

        print("# time-dataset: ", len(self.time_dataset))

        print(
            f"""Yelp Dataset(T={len(times)},metadata={self.metadata},dataset={dataset},
            )"""
        )

    @property
    def test_dataset(self):
        data = [(self.time_dataset, self.eval_datas[2])]
        return data

    @property
    def val_dataset(self):
        data = [(self.time_dataset, self.eval_datas[1])]
        return data

    @property
    def train_dataset(self):
        data = [(self.time_dataset, self.eval_datas[0])]
        return data

    def to(self, device):
        self.device = device
        self.dataset = move_to(self.dataset, self.device)
        self.time_dataset = move_to(self.time_dataset, device)
        self.eval_datas = move_to(self.eval_datas, device)


if __name__ == "__main__":
    dataset = YelpDataset()
    dataset.dataset
