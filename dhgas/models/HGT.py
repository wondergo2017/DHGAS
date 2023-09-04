import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math
from torch_geometric.nn import Linear


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=240, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, t):
        return self.lin(self.emb(t))


class HGT(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_heads,
        num_layers,
        metadata,
        predict_type,
        use_RTE=False,
        featemb=None,
        nclf_linear=None,
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types = metadata[0]
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                hidden_channels,
                hidden_channels,
                metadata,
                num_heads,
                group="sum",
                use_RTE=use_RTE,
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

        self.predict_type = predict_type

        self.featemb = featemb if featemb else lambda x: x

        self.nclf_linear = nclf_linear

        print(f"use RTE : {use_RTE}")

    def forward(self, x_dict, edge_index_dict, edge_time_dict=None):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()  # use tanh

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_time_dict)

        out = x_dict
        predict_type = self.predict_type
        if isinstance(predict_type, list):
            out = [self.lin(out[predict_type[0]]), self.lin(out[predict_type[1]])]
        else:
            out = self.lin(out[predict_type])

        return out

    def encode(self, data, *args, **kwargs):
        x = self.featemb(data.x_dict)
        e = data.edge_index_dict
        if hasattr(data, "edge_time_dict"):
            et = data.edge_time_dict
        else:
            et = None
        return self.forward(x, e, et)

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf_linear(z)
        return out


from typing import Union, Dict, Optional, List
from torch_geometric.typing import NodeType, EdgeType, Metadata

import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, ones, reset


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using HGT, see `examples/hetero/hgt_dblp.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/hgt_dblp.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        group (string, optional): The aggregation scheme to use for grouping
            node embeddings generated by different relations.
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        group: str = "sum",
        use_RTE=False,
        **kwargs,
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.group = group
        self.rte = RelTemporalEncoding(out_channels) if use_RTE else None
        self.use_RTE = use_RTE
        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        self.skip = torch.nn.ParameterDict()
        for node_type, in_channels in self.in_channels.items():
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(out_channels, out_channels)
            self.skip[node_type] = Parameter(torch.Tensor(1))

        self.a_rel = torch.nn.ParameterDict()
        self.m_rel = torch.nn.ParameterDict()
        self.p_rel = torch.nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = "__".join(edge_type)
            self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.p_rel[edge_type] = Parameter(torch.Tensor(heads))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor], Dict[EdgeType, SparseTensor]],
        edge_time_dict=None,
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict: (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The ouput node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """

        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Iterate over node-types:
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
            q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
            v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if self.use_RTE:
                edge_time = edge_time_dict[edge_type].squeeze(-1)
            edge_type = "__".join(edge_type)
            a_rel = self.a_rel[edge_type]
            m_rel = self.m_rel[edge_type]

            time_emb_k = None
            time_emb_v = None
            if self.use_RTE:
                time_emb = self.rte(edge_time)
                time_emb_k = (
                    self.k_lin[src_type](time_emb).view(-1, H, D).transpose(0, 1)
                    @ a_rel
                ).transpose(1, 0)
                time_emb_v = (
                    self.v_lin[src_type](time_emb).view(-1, H, D).transpose(0, 1)
                    @ m_rel
                ).transpose(1, 0)

            k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)
            v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)

            # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
            out = self.propagate(
                edge_index,
                edge_time_k=time_emb_k,
                edge_time_v=time_emb_v,
                k=k,
                q=q_dict[dst_type],
                v=v,
                rel=self.p_rel[edge_type],
                size=None,
            )
            out_dict[dst_type].append(out)

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            out = group(outs, self.group)

            if out is None:
                out_dict[node_type] = None
                continue

            out = self.a_lin[node_type](F.gelu(out))
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(
        self,
        k_j: Tensor,
        q_i: Tensor,
        v_j: Tensor,
        rel: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
        edge_time_k,
        edge_time_v,
    ) -> Tensor:
        if self.use_RTE:
            k_j = k_j + edge_time_k
            v_j = v_j + edge_time_v
        alpha = (q_i * k_j).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.out_channels}, " f"heads={self.heads})"
