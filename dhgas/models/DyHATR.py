from .HLinear import HLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math
from torch_geometric.nn import GATConv


class RelationAgg(nn.Module):
    def __init__(self, n_inp: int, n_hid: int):
        super(RelationAgg, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(n_inp, n_hid), nn.Tanh(), nn.Linear(n_hid, 1, bias=False)
        )

    def forward(self, h):
        w = self.project(h).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)

        return (beta * h).sum(1)


class TemporalAttentionLayer(nn.Module):
    def __init__(
        self, input_dim, n_heads, num_time_steps, attn_drop=0, residual=False
    ):  # default setting in original codes
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        position_inputs = (
            torch.arange(0, self.num_time_steps)
            .reshape(1, -1)
            .repeat(inputs.shape[0], 1)
            .long()
            .to(inputs.device)
        )
        temporal_inputs = (
            inputs + self.position_embeddings[position_inputs]
        )  # [N, T, F]
        # temporal_inputs = inputs

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(
            temporal_inputs, self.Q_embedding_weights, dims=([2], [0])
        )  # [N, T, F]
        k = torch.tensordot(
            temporal_inputs, self.K_embedding_weights, dims=([2], [0])
        )  # [N, T, F]
        v = torch.tensordot(
            temporal_inputs, self.V_embedding_weights, dims=([2], [0])
        )  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        # [hN, T, F/h]
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)
        # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)
        # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps**0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-(2**32) + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(
            torch.split(
                outputs,
                split_size_or_sections=int(outputs.shape[0] / self.n_heads),
                dim=0,
            ),
            dim=2,
        )  # [N, T, F]

        # 6: Feedforward and residual
        # outputs = self.feedforward(outputs)
        # if self.residual:
        #     outputs = outputs + temporal_inputs
        # outputs = outputs +inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)


class DyHATRLayer(nn.Module):
    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        n_heads: int,
        timeframe: list,
        dropout: float,
        metadata,
        edge_layers=1,
    ):
        super(DyHATRLayer, self).__init__()

        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_heads = n_heads
        self.timeframe = timeframe
        self.dropout = dropout
        self.ntypes, self.etypes = metadata
        self.edge_layers = edge_layers

        self.rnn_layer = 1  # use only 1 layer in original codes
        # intra reltion aggregation modules
        self.intra_rel_agg = nn.ModuleDict(
            {
                etype: nn.ModuleList(
                    [
                        GATConv(
                            n_inp,
                            n_hid,
                            n_heads,
                            dropout=dropout,
                            concat=False,
                            bias=False,
                        )
                        for i in range(edge_layers)
                    ]
                )  # without bias and concat in original code
                for _, etype, _ in self.etypes
            }
        )

        # inter relation aggregation modules
        self.inter_rel_agg = RelationAgg(
            n_hid, 4 * n_hid
        )  # in original code, hidden=32, and attn_vec=128. Similarly, we use attn_vec=4*n_hid

        # inter time aggregation modules
        self.rnn = nn.LSTM(
            n_hid, n_hid, self.rnn_layer, batch_first=True
        )  # use best performed variant in original paper , LSTM instead of GRU
        self.timeattn = TemporalAttentionLayer(
            n_hid, n_heads, len(self.timeframe), 0, False
        )  # dropout=0,residual=False in original codes

    def forward(self, graphs, node_features):
        # node_features # [time,ntype,N,F]

        # edge-specific
        intra_features = dict(
            {ttype: {} for ttype in self.timeframe}
        )  # [ttype,etype] -> [N,F]
        for ttype in self.timeframe:
            graph = graphs[ttype]
            x = node_features[ttype]  #  [ntype,N,F]
            for ntype1, etype, ntype2 in self.etypes:
                edge_index = graph[etype].edge_index
                xi = x[ntype1]
                xj = x[ntype2]
                for i in range(self.edge_layers):
                    out = self.intra_rel_agg[etype][i]((xi, xj), edge_index)
                    if i != self.edge_layers - 1:
                        out = F.relu(out)
                intra_features[ttype][etype] = out

        # edge
        inter_features = dict(
            {ntype: {} for ntype in self.ntypes}
        )  # [ntype,ttype] -> [N,F]
        for ttype in intra_features.keys():
            for ntype in self.ntypes:
                types_features = []
                for stype, etype, dtype in self.etypes:
                    if ntype == dtype:
                        type_feature = intra_features[ttype][etype]
                        types_features.append(type_feature)
                types_features = torch.stack(types_features, dim=1)  # [N,etypes,F]
                out_feat = self.inter_rel_agg(types_features)
                inter_features[ntype][ttype] = out_feat
        # time
        output_features = dict(
            {ttype: {} for ttype in self.timeframe}
        )  # [time,ntype,N,F]

        def timeaggr(feats):
            h0 = torch.zeros((self.rnn_layer, feats.shape[0], self.n_hid)).to(
                feats.device
            )
            c0 = torch.zeros((self.rnn_layer, feats.shape[0], self.n_hid)).to(
                feats.device
            )
            feats, _ = self.rnn(feats, (h0, c0))  # [N,T,F]
            feats = self.timeattn(feats)
            return feats

        for ntype in inter_features:
            out_emb = [inter_features[ntype][ttype] for ttype in inter_features[ntype]]
            time_embeddings = torch.stack(out_emb, dim=1)  # [N,T,F]
            h = timeaggr(time_embeddings)  # [N,T,F]
            for i, ttype in enumerate(self.timeframe):
                output_features[ttype][ntype] = h[:, i, :]

        return output_features


class DyHATR(nn.Module):
    def __init__(
        self,
        in_dim,
        n_hid: int,
        n_layers: int,
        n_heads: int,
        time_window: int,
        metadata,
        predict_type,
        dropout: float = 0.2,
        edge_layers=2,
        featemb=None,
        nclf_linear=None,
    ):
        super(DyHATR, self).__init__()
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.metadata = metadata
        self.predict_type = predict_type
        self.timeframe = list(range(time_window))

        self.gnn_layers = nn.ModuleList(
            [
                DyHATRLayer(
                    in_dim,
                    n_hid,
                    n_heads,
                    self.timeframe,
                    dropout,
                    metadata=metadata,
                    edge_layers=edge_layers,
                )
                for _ in range(n_layers)
            ]
        )
        # self.hlinear = HLinear(n_hid, metadata, act='tanh') # check tanh
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

        out = inp_feat[self.timeframe[-1]]

        predict_type = self.predict_type
        if isinstance(predict_type, list):
            out = [out[predict_type[0]], out[predict_type[1]]]
        else:
            out = out[predict_type]
        # out_feat = F.normalize(out_feat, p=2, dim=-1,eps=1e-12) # use this in original code
        return out

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf_linear(z)
        return out
