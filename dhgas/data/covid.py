import os
import os.path as osp
import random
import torch
from torch_geometric.data import HeteroData
from dhgas.utils import move_to, timeit
from .utils import *
from torch_geometric.data import HeteroData
from dhgas import config
from ..utils import setup_seed

dataroot = osp.join(config.dataroot, "covid/")
process_dir = osp.join(dataroot, "processed/")


class CovidDataset:
    def __init__(
        self,
    ):
        processed = osp.join(process_dir, f"covid.pt")
        if osp.exists(processed):
            print(f"loading {processed}")
            dataset = torch.load(processed)
        else:
            os.makedirs(process_dir, exist_ok=True)
            dataset = self.preprocess()
            print(f"saving {processed}")
            torch.save(dataset, processed)

        self.dataset = dataset

    def preprocess(self):
        from dgl.data.utils import load_graphs

        glist, _ = load_graphs(osp.join(dataroot, "covid_graphs.bin"))
        datas = []
        for g in glist:
            data = HeteroData()
            for ntype in "state county".split():
                data[ntype].x = g.nodes[ntype].data["feat"]
            for stype, etype, ttype in g.canonical_etypes:
                # src, dst = g.in_edges(g.nodes(ttype), etype=etype)
                # data[(stype,etype,ttype)].edge_index=torch.stack([src,dst]).long()
                data[(stype, etype, ttype)].edge_index = torch.stack(
                    g.edges(etype=etype)
                ).long()
            datas.append(data)
        return datas

    def times(self):
        return list(range(len(self.dataset)))


from torch_geometric import seed_everything


class CovidUniDataset:
    def __init__(
        self, time_window=1, shuffle=True, test_full=False, is_dynamic=True, seed=22
    ):
        self.time_window = time_window
        self.shuffle = shuffle
        self.is_dynamic = is_dynamic

        setup_seed(seed)  # seed preprocess
        dataset = CovidDataset()

        setup_seed(seed)  # seed spliting
        times = dataset.times()  # [0-303]
        dataset = dataset.dataset

        self.dataset = dataset
        self.metadata = dataset[0].metadata()
        maxt = len(times)
        print(f"time length :  {maxt}")
        self.time_dataset = {}
        for split in "train val test".split():
            self.time_dataset[split] = []

        for i in times:
            if i < time_window:
                continue
            if is_dynamic:
                support = dataset[i - time_window : i]
            else:
                support = dataset[i - 1]
            query = dataset[i]["state"]
            if i >= maxt - 30:
                split = "test"
            elif i >= maxt - 60:
                split = "val"
            else:
                split = "train"
            self.time_dataset[split].append((support, query))

        for split in "train val test".split():
            print(f"#split {split} = {len(self.time_dataset[split])}")

    @property
    def test_dataset(self):
        return self.time_dataset["test"]

    @property
    def val_dataset(self):
        return self.time_dataset["val"]

    @property
    def train_dataset(self):
        data = self.time_dataset["train"]
        if self.shuffle:
            random.shuffle(data)
        return data

    def to(self, device):
        self.time_dataset = move_to(self.time_dataset, device)


if __name__ == "__main__":
    covid = CovidDataset()
    covid.dataset

    covid = CovidUniDataset(time_window=3, is_dynamic=True)

    covid.dataset
