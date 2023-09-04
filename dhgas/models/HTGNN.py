import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import GATConv


class RelationAgg(nn.Module):
    def __init__(self, n_inp: int, n_hid: int):
        """

        :param n_inp: int, input dimension
        :param n_hid: int, hidden dimension
        """
        super(RelationAgg, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(n_inp, n_hid), nn.Tanh(), nn.Linear(n_hid, 1, bias=False)
        )

    def forward(self, h):
        w = self.project(h).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)

        return (beta * h).sum(1)


class TemporalAgg(nn.Module):
    def __init__(self, n_inp: int, n_hid: int, time_window: int, device: torch.device):
        """

        :param n_inp      : int         , input dimension
        :param n_hid      : int         , hidden dimension
        :param time_window: int         , the number of timestamps
        :param device     : torch.device, gpu
        """
        super(TemporalAgg, self).__init__()

        self.proj = nn.Linear(n_inp, n_hid)
        self.q_w = nn.Linear(n_hid, n_hid, bias=False)
        self.k_w = nn.Linear(n_hid, n_hid, bias=False)
        self.v_w = nn.Linear(n_hid, n_hid, bias=False)
        self.fc = nn.Linear(n_hid, n_hid)
        self.pe = torch.tensor(
            self.generate_positional_encoding(n_hid, time_window)
        ).float()

    def generate_positional_encoding(self, d_model, max_len):
        pe = np.zeros((max_len, d_model))
        for i in range(max_len):
            for k in range(0, d_model, 2):
                div_term = math.exp(k * -math.log(100000.0) / d_model)
                pe[i][k] = math.sin((i + 1) * div_term)
                try:
                    pe[i][k + 1] = math.cos((i + 1) * div_term)
                except:
                    continue
        return pe

    def forward(self, x):  # x [N,T,F]
        h = self.proj(x)
        h = h + self.pe.to(h.device)
        q = self.q_w(h)
        k = self.k_w(h)
        v = self.v_w(h)

        qk = torch.matmul(q, k.permute(0, 2, 1))  # [N,T,T]
        score = F.softmax(qk, dim=-1)  # [N,T,T]

        h_ = torch.matmul(score, v)
        h_ = F.relu(self.fc(h_))

        return h_  # [N,T,F]


class HTGNNLayer(nn.Module):
    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        n_heads: int,
        timeframe: list,
        norm: bool,
        device: torch.device,
        dropout: float,
        metadata,
    ):
        super(HTGNNLayer, self).__init__()

        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_heads = n_heads
        self.timeframe = timeframe
        self.norm = norm
        self.dropout = dropout
        self.ntypes, self.etypes = metadata

        # intra reltion aggregation modules
        self.intra_rel_agg = nn.ModuleDict(
            {
                etype: GATConv(
                    n_inp, n_hid, n_heads, dropout=dropout, concat=False, bias=True
                )
                for _, etype, _ in self.etypes
            }
        )

        # inter relation aggregation modules
        self.inter_rel_agg = nn.ModuleList(
            {RelationAgg(n_hid, n_hid) for ttype in timeframe}
        )

        # inter time aggregation modules
        # tattn=
        self.cross_time_agg = nn.ModuleDict(
            {
                ntype: TemporalAgg(n_hid, n_hid, len(timeframe), device)
                for ntype in self.ntypes
            }
        )

        # gate mechanism
        self.res_fc = nn.ModuleDict()
        self.res_weight = nn.ParameterDict()
        for ntype in self.ntypes:
            self.res_fc[ntype] = nn.Linear(n_inp, n_hid)  # n_heads*n_hid
            self.res_weight[ntype] = nn.Parameter(torch.randn(1))

        self.reset_parameters()

        # LayerNorm
        if norm:
            self.norm_layer = nn.ModuleDict(
                {ntype: nn.LayerNorm(n_hid) for ntype in self.ntypes}
            )

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        gain = nn.init.calculate_gain("relu")
        for ntype in self.res_fc:
            nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)

    def forward(self, graphs, node_features):
        # edge-specific
        # [ttype,etype] -> [N,F]
        intra_features = dict({ttype: {} for ttype in self.timeframe})
        for ttype in self.timeframe:
            graph = graphs[ttype]
            x = node_features[ttype]  # [ntype,N,F]
            for ntype1, etype, ntype2 in self.etypes:
                edge_index = graph[etype].edge_index
                xi = x[ntype1]
                xj = x[ntype2]
                out = self.intra_rel_agg[etype]((xi, xj), edge_index)
                intra_features[ttype][etype] = out

        # edge
        # [ntype,ttype] -> [N,F]
        inter_features = dict({ntype: {} for ntype in self.ntypes})
        for ttype in self.timeframe:
            for ntype in self.ntypes:
                types_features = []
                for stype, etype, dtype in self.etypes:
                    if ntype == dtype:
                        type_feature = intra_features[ttype][etype]
                        types_features.append(type_feature)
                types_features = torch.stack(types_features, dim=1)  # [N,etypes,F]
                out_feat = self.inter_rel_agg[ttype](types_features)
                inter_features[ntype][ttype] = out_feat
        # time
        output_features = dict(
            {ttype: {} for ttype in self.timeframe}
        )  # [time,ntype,N,F]

        for ntype in self.ntypes:
            out_emb = [inter_features[ntype][ttype] for ttype in inter_features[ntype]]
            time_embeddings = torch.stack(out_emb, dim=1)  # [N,T,F]
            h = self.cross_time_agg[ntype](time_embeddings)  # [N,T,F]
            for i, ttype in enumerate(self.timeframe):
                output_features[ttype][ntype] = h[:, i, :]

        # update
        new_features = dict({ttype: {} for ttype in self.timeframe})  # [time,ntype,N,F]
        for ntype in self.ntypes:
            alpha = torch.sigmoid(self.res_weight[ntype])
            for ttype in self.timeframe:
                new_features[ttype][ntype] = output_features[ttype][
                    ntype
                ] * alpha + self.res_fc[ntype](node_features[ttype][ntype]) * (
                    1 - alpha
                )
                if self.norm:
                    new_features[ttype][ntype] = self.norm_layer[ntype](
                        new_features[ttype][ntype]
                    )

        return new_features  # [time,ntype,N,F]


from dhgas.models.HLinear import HLinear


class HTGNN(nn.Module):
    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        n_layers: int,
        n_heads: int,
        time_window: int,
        norm: bool,
        device: torch.device,
        metadata,
        predict_type,
        dropout: float = 0.2,
        featemb=None,
        nclf_linear=None,
    ):
        super(HTGNN, self).__init__()

        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.metadata = metadata
        self.timeframe = list(range(time_window))
        self.predict_type = predict_type
        self.gnn_layers = nn.ModuleList(
            [
                HTGNNLayer(
                    n_hid,
                    n_hid,
                    n_heads,
                    self.timeframe,
                    norm,
                    device,
                    dropout,
                    metadata=metadata,
                )
                for _ in range(n_layers)
            ]
        )
        # self.hlinear = HLinear(n_hid, metadata, act='tanh')
        self.hlinear = HLinear(n_hid, metadata, act="None")

        self.featemb = featemb if featemb else lambda x: x
        self.nclf_linear = nclf_linear

    def encode(self, graphs):
        inp_feat = {}  # [time,ntype,N,F]
        for ttype in self.timeframe:
            graph = graphs[ttype]
            x_dict = self.featemb(graph.x_dict)
            x_dict = self.hlinear(x_dict)
            inp_feat[ttype] = x_dict

        for i in range(self.n_layers):
            inp_feat = self.gnn_layers[i](graphs, inp_feat)

        predict_type = self.predict_type
        if isinstance(predict_type, list):
            out_feat0 = sum(
                [inp_feat[ttype][self.predict_type[0]] for ttype in self.timeframe]
            )
            out_feat1 = sum(
                [inp_feat[ttype][self.predict_type[1]] for ttype in self.timeframe]
            )
            out_feat = [out_feat0, out_feat1]
        else:
            out_feat = sum(
                [inp_feat[ttype][self.predict_type] for ttype in self.timeframe]
            )

        return out_feat

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf_linear(z)
        return out
