from torch_geometric.utils import contains_self_loops, is_undirected
import torch
from copy import deepcopy
from gensim.models import Word2Vec
from torch_geometric.data import HeteroData, Data
import gensim
from torch_geometric.utils import negative_sampling
import torch as th
from torch_geometric.data import HeteroData
from collections import Counter
import numpy as np


def setorderidx(data):
    """map col idx to ordered idx starting from 0
    @params data: numpy array , rows as samples, columns as attributes
    @return data: numpy array
    """
    data = data.copy()
    row, col = data.shape
    cnt = {}
    for i in range(col):
        cnt[i] = Counter(data[:, i])
        k = list(cnt[i].keys())
        k.sort()
        k2i = dict(zip(k, range(len(k))))
        # print(f'mapping col {i} head 50')
        # print(list(k2i.items())[:50])
        for j in range(row):
            data[j][i] = k2i[data[j][i]]
    data = np.vectorize(int)(data)
    return data


# @timeit
def time_select_edge_time(dataset, t):
    """select by time t. time is stored in edge_attr in Long Tensor
    @param t: time index
    @return : HeteroData , with same nodes ,with sliced edgeindex and edge attr
    """
    dt = HeteroData()
    d = dataset
    for ntype, value in d.x_dict.items():
        dt[ntype].x = value
        if "num_nodes" in d[ntype].keys():
            dt[ntype].num_nodes = d[ntype].num_nodes
    dea = d.edge_time_dict
    dei = d.edge_index_dict
    for etype in dea:
        mask = (dea[etype] == t).squeeze(-1)
        dt[etype].edge_index = dei[etype][:, mask]
        dt[etype].edge_time = dea[etype][mask]
    return dt


def time_merge_edge_time(datalist):
    """merge dataset selected by time. time is stored in edge_attr in Long Tensor
    @params datalist : list of HeteroData
    @return : merged HeteroData
    """
    d = datalist
    d0 = d[0]
    dt = HeteroData()
    for ntype, value in d0.x_dict.items():
        dt[ntype].x = value
    edge_index_dict = {}
    edge_time_dict = {}
    for etype in d0.edge_time_dict:
        for i in range(len(d)):
            edge_index_dict[etype] = edge_index_dict.get(etype, [])
            edge_index_dict[etype].append(d[i].edge_index_dict[etype])
            edge_time_dict[etype] = edge_time_dict.get(etype, [])
            edge_time_dict[etype].append(d[i].edge_time_dict[etype])
        dt[etype].edge_index = th.cat(edge_index_dict[etype], dim=1)
        dt[etype].edge_time = th.cat(edge_time_dict[etype], dim=0)
    return dt


from dhgas.utils import timeit

# @timeit
def linksplit(data, all_neg=False, inplace=False):
    """HomoData, do negative sampling
    @params data: HomoData
    @return : HomoData
    """
    if not inplace:
        data = data.clone()
    device = data.edge_index.device
    ei = data.edge_index.to("cpu").numpy()

    # reorder
    nodes = list(set(ei.flatten()))
    nodes.sort()
    id2n = nodes
    n2id = dict(zip(nodes, np.arange(len(nodes))))

    # construct the graph containing these nodes
    ei_ = np.vectorize(lambda x: n2id[x])(ei)

    if all_neg:
        maxn = len(nodes)
        nei_ = []
        pos_e = set([tuple(x) for x in ei_.T])
        for i in range(maxn):
            for j in range(maxn):
                if i != j and (i, j) not in pos_e:
                    nei_.append([i, j])
        nei_ = torch.LongTensor(nei_).T
    else:
        nei_ = negative_sampling(th.LongTensor(ei_))
    nei = th.LongTensor(np.vectorize(lambda x: id2n[x])(nei_.numpy()))
    ei = th.LongTensor(ei)

    # add to edge attr
    data.edge_label_index = th.cat([ei, nei], dim=-1).to(device)
    data.edge_label = th.cat([th.ones(ei.shape[1]), th.zeros(nei.shape[1])], dim=-1).to(
        device
    )
    return data


def hetero_linksplit(data, etype, inplace=False):
    """hetero_data, do negative sampling
    @params data: hetero_data
    @return : hetero_data
    """
    if not inplace:
        data = data.clone()
    device = data[etype].edge_index.device
    ei = data[etype].edge_index.to("cpu").numpy()

    # reorder
    nodes0 = list(set(ei[0].flatten()))
    nodes0.sort()
    nodes1 = list(set(ei[1].flatten()))
    nodes1.sort()

    id2n0 = nodes0
    id2n1 = nodes1
    n02id = dict(zip(nodes0, np.arange(len(nodes0))))
    n12id = dict(zip(nodes1, np.arange(len(nodes1))))
    size = (len(nodes0), len(nodes1))
    # construct the graph containing these nodes
    ei_ = np.apply_along_axis(lambda x: (n02id[x[0]], n12id[x[1]]), axis=0, arr=ei)
    nei_ = negative_sampling(th.LongTensor(ei_), size).numpy()
    nei = th.LongTensor(
        np.apply_along_axis(lambda x: (id2n0[x[0]], id2n1[x[1]]), axis=0, arr=nei_)
    )
    ei = th.LongTensor(ei)

    # add to edge attr
    data[etype].edge_label_index = th.cat([ei, nei], dim=-1).to(device)
    data[etype].edge_label = th.cat(
        [th.ones(ei.shape[1]), th.zeros(nei.shape[1])], dim=-1
    ).to(device)
    return data


def shift_negative_sample(data, shift):
    """HomoData, efficient negative sampling for bipartite graph
    @params data: HomoData
    @params shift: index shift
    """
    ei = data.edge_index
    nei = negative_sampling(ei - shift.to(ei.device))
    nei = nei.to(ei.device) + shift.to(ei.device)
    # add to edge attr
    data.edge_label_index = th.cat([ei, nei], dim=-1)
    data.edge_label = th.cat([th.ones(ei.shape[1]), th.zeros(nei.shape[1])], dim=-1).to(
        ei.device
    )

    return data


def negative_sample(data):
    ei = data.edge_index
    nei = negative_sampling(ei)
    nei = nei.to(ei.device)
    # add to edge attr
    data.edge_label_index = th.cat([ei, nei], dim=-1)
    data.edge_label = th.cat([th.ones(ei.shape[1]), th.zeros(nei.shape[1])], dim=-1).to(
        ei.device
    )
    return data


def sorteddict(x, min=True):
    """return dict sorted by values
    @params x: a dict
    @params min : whether from small to large.
    """
    if min:
        return dict(sorted(x.items(), key=lambda item: item[1]))
    else:
        return dict(sorted(x.items(), key=lambda item: item[1])[::-1])


def sen2vec(sentences, vector_size=32):
    """use gensim.word2vec to generate wordvecs, and average as sentence vectors.
    if exception happens use zero embedding.
    @ params sentences : list of sentence
    @ params vector_size
    @ return : sentence embedding
    """
    sentences = [list(gensim.utils.tokenize(a, lower=True)) for a in sentences]
    sentences
    vector_size = 32
    model = Word2Vec(sentences, vector_size=vector_size, min_count=1)
    print("word2vec done")
    embs = []
    for s in sentences:
        try:
            emb = model.wv[s]
            emb = np.mean(emb, axis=0)
        except Exception as e:
            print(e)
            emb = np.zeros(vector_size)
        embs.append(emb)
    embs = np.stack(embs)
    print(f"emb shape : {embs.shape}")
    return embs


def map2id(l):
    """encode list (unique) starting from 0.
    @return : mapping dict from l -> int.
    """
    return dict(zip(l, range(len(l))))


def flip_edge_index(edge_index_dict):
    """turn src to dst, dst to src"""
    edge_index = deepcopy(edge_index_dict)
    for etype in edge_index:
        e = edge_index[etype]
        e = th.stack([e[1, :], e[0, :]], dim=0)
        edge_index[etype] = e
    return edge_index


def mask2idx(mask):
    """turn [0,1,1,0,1] to [1,2,4]
    @params mask : (N,)
    @return : list
    """
    idxs = []
    for i, v in enumerate(mask):
        if v:
            idxs.append(i)
    return idxs


def select_edge_index_by_idxs(edge_index, idxs, src=True):
    """
    @params edge_index: data.edge_index Tensor([2,E])
    @params idxs : [1,2,5,8,...]
    @params src : slice based on src, else dst nodes.
    """
    edges = []

    idxs = set(idxs)
    for e in edge_index.T.numpy():
        node = e[0] if src else e[1]
        if node in idxs:
            edges.append(e)
    return th.LongTensor(edges).T


def time_select_node_attr(dataset, t, attr="time"):
    """select by time t. time is stored in node by named ``attr``
    @param t: time index
    @param attr:
    @return : HeteroData , with same nodes ,with sliced edgeindex
    """
    dt = HeteroData()
    d = dataset
    for ntype, value in d.x_dict.items():
        dt[ntype].x = value
    time_dict = getattr(d, f"{attr}_dict")
    time_node_types = [nt for nt in time_dict]
    dei = d.edge_index_dict
    for nt in time_node_types:
        setattr(dt[nt], attr, time_dict[nt])

        year = time_dict[nt].squeeze(-1)
        mask = (year == t).numpy()
        idxs = mask2idx(mask)

        for etype in dei:
            src, rel, dst = etype
            if src == nt:
                dt[etype].edge_index = select_edge_index_by_idxs(
                    dei[etype], idxs, src=True
                )
            elif dst == nt:
                dt[etype].edge_index = select_edge_index_by_idxs(
                    dei[etype], idxs, src=False
                )
            else:
                dt[etype].edge_index = dei[etype]
    return dt


def time_merge_node_attr(datalist, attr="time"):
    """merge dataset selected by time. time is stored in node by named ``attr``
    @params datalist : list of HeteroData
    @params attr:
    @return : merged HeteroData
    """
    d = datalist
    d0 = d[0]
    dt = HeteroData()
    for ntype, value in d0.x_dict.items():
        dt[ntype].x = value
    time_dict = getattr(d0, f"{attr}_dict")
    time_node_types = [nt for nt in time_dict]

    for nt in time_node_types:
        setattr(dt[nt], attr, time_dict[nt])

    edge_index_dict = {}
    for etype in d0.edge_index_dict:
        for i in range(len(d)):
            edge_index_dict[etype] = edge_index_dict.get(etype, [])
            edge_index_dict[etype].append(d[i].edge_index_dict[etype])
        dt[etype].edge_index = th.cat(edge_index_dict[etype], dim=1)

    if hasattr(d[0], "edge_time_dict"):
        edge_time_dict = {}
        for etype in d0.edge_time_dict:
            for i in range(len(d)):
                edge_time_dict[etype] = edge_time_dict.get(etype, [])
                edge_time_dict[etype].append(d[i].edge_time_dict[etype])
            dt[etype].edge_time = th.cat(edge_time_dict[etype], dim=1)
    return dt


def get_index_map_homo2hetero(data):
    """
    @params data: HeteroData
    @return (
        node_ids , dict # type : local ->  global
        index_map , list # global -> local
        node_type_names, list # type id -> name
    )
    """
    node_type = data.node_type
    node_type_names = data._node_type_names
    node_ids, index_map = {}, torch.empty_like(node_type)
    for i, key in enumerate(node_type_names):
        node_ids[i] = (node_type == i).nonzero(as_tuple=False).view(-1)
        index_map[node_ids[i]] = torch.arange(len(node_ids[i]))
    return node_ids, index_map, node_type_names


def to_homo_coauthor(homodata, coauthor_data):
    """
    @params homodata : homodata from heteroData
    @params coauthor_data : coauthor_data with local indexs in heteroData
    """
    coauthor_data = coauthor_data.clone()
    local2global, global2local, node_type_names = get_index_map_homo2hetero(homodata)
    local_edges = coauthor_data.edge_index
    global_edges = (
        local2global[1]
        .index_select(0, local_edges.flatten())
        .reshape(local_edges.shape)
    )
    coauthor_data.edge_index = global_edges
    return coauthor_data


def get_author_graph(data):
    """get coauthor graph
    @params data : HeteroData
    @return coauthor graph: pyg.Data
    """
    es = data["written"].edge_index.numpy().T

    # e1 is paper , e2 is author
    p2a = {}
    for (e1, e2) in es:
        if e1 not in p2a:
            p2a[e1] = set()
        p2a[e1].add(e2)
    p2a

    # author graph does not include self-edge
    author_edges = []
    for p, authors in p2a.items():
        authors = list(authors)
        na = len(authors)
        for i in range(na):
            for j in range(i + 1, na):
                author_edges.append([authors[i], authors[j]])
                author_edges.append([authors[j], authors[i]])
    author_edges = torch.LongTensor(np.array(author_edges).T)
    data = Data(edge_index=author_edges)
    return data


def num_coin_edges(pos_edges, neg_edges):
    """number of same edges in 2 ajacency matrix"""
    return len(
        set(tuple(x.numpy()) for x in pos_edges.T)
        & set(tuple(x.numpy()) for x in neg_edges.T)
    )


def check_link_split(coauthor_graph):
    """check some basic properties of graph and linksplit."""
    d = linksplit(coauthor_graph)
    print(
        f"original graph self loops :", contains_self_loops(d.edge_index)
    )  # check self-loops in the original edge_index
    print(f"undirected : {is_undirected(d.edge_index)}")

    pos_edges = d.edge_label_index[:, d.edge_label.bool()]
    neg_edges = d.edge_label_index[:, ~d.edge_label.bool()]

    print(
        "pos edges are unchagned:", (pos_edges == d.edge_index).all()
    )  # check pos edges
    print(
        "neg edges has self loops", contains_self_loops(neg_edges)
    )  # check self-loops in the negative edge_index

    def num_self_loops(edge_index):

        mask = edge_index[0] == edge_index[1]
        return mask.sum().item()

    print("neg edges have loops :", num_self_loops(neg_edges))
    print(
        "coincidence of neg edges and pos edges", num_coin_edges(pos_edges, neg_edges)
    )


def select_x(data, ntype):
    return data.x[data.node_type == ntype]


def select_edge_index(data, etype):
    return data.edge_index[:, data.edge_type == etype]


def make_hodata(x_dict, e_dict, predict_type):
    hodata = HeteroData()
    for ntype, x in x_dict.items():
        hodata[ntype].x = x
    for etype, e in e_dict.items():
        hodata[etype].edge_index = e
    hodata = hodata.to_homogeneous()
    node_type = hodata.node_type
    node_type_names = hodata._node_type_names
    name2id = dict(zip(node_type_names, range(len(node_type_names))))
    if isinstance(predict_type, list) or isinstance(predict_type, tuple):
        mask0 = node_type == name2id[predict_type[0]]
        mask1 = node_type == name2id[predict_type[1]]
        mask = [mask0, mask1]
    else:
        mask = node_type == name2id[predict_type]
    x = hodata.x
    e = hodata.edge_index
    return x, e, mask, hodata
