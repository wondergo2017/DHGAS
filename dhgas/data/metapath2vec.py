from torch_geometric.nn import MetaPath2Vec
from tqdm import tqdm
import torch
import numpy as np


def get_metapath2vec(data, metapath, epochs=1, emb_size=32, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    num_nodes_dict = data.num_nodes_dict
    model = MetaPath2Vec(
        data.edge_index_dict,
        embedding_dim=emb_size,
        metapath=metapath,
        walk_length=50,
        context_size=7,
        walks_per_node=5,
        num_negative_samples=5,
        sparse=True,
        num_nodes_dict=num_nodes_dict,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=6)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train(epoch):
        model.train()
        total_loss = []
        for i, (pos_rw, neg_rw) in enumerate(tqdm(loader)):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
        avgloss = np.mean(total_loss)
        print("loss:", avgloss)
        return avgloss

    for i in range(epochs):
        train(i)
    nodevecs = {}
    with torch.no_grad():
        for n in data.node_types:
            nodevecs[n] = model(n).to("cpu").detach()
    return nodevecs
