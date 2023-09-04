import torch
from torch_geometric import nn as nn
from torch_geometric.nn import GCNConv
from dhgas.data.utils import make_hodata


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layers,
        metadata,
        predict_type,
        featemb=None,
        nclf_linear=None,
    ):
        super().__init__()
        convs = torch.nn.ModuleList()
        convs.append(GCNConv(in_dim, hid_dim))
        for i in range(num_layers - 1):
            convs.append(GCNConv(hid_dim, hid_dim))
        self.convs = convs
        # self.hlinear = HLinear(hid_dim, metadata, act='None')
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x
        self.nclf = nclf_linear

    def encode(self, data, *args, **kwargs):
        x_dict = self.featemb(data.x_dict)
        # x_dict = self.hlinear(x_dict)
        e_dict = data.edge_index_dict

        x, e, mask, _ = make_hodata(x_dict, e_dict, self.predict_type)

        for i, conv in enumerate(self.convs):
            x = conv(x, e)
            if i != len(self.convs) - 1:
                x = x.relu()

        if isinstance(mask, list):
            x = [x[mask[0]], x[mask[1]]]
        else:
            x = x[mask]
        return x

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf(z)
        return out
