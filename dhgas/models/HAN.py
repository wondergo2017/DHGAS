import torch
import torch.nn as nn
from torch_geometric.nn import HANConv


class HAN(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        num_layers,
        metadata,
        predict_type,
        heads=8,
        dropout=0.6,
        featemb=None,
        nclf_linear=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HANConv(
                -1, hidden_channels, heads=heads, dropout=dropout, metadata=metadata
            )
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x

        self.nclf = nclf_linear

    def forward(self, x_dict, edge_index_dict):
        out = x_dict
        for i in range(self.num_layers):
            out = self.convs[i](out, edge_index_dict)

        predict_type = self.predict_type
        if isinstance(predict_type, list):
            out = [self.lin(out[predict_type[0]]), self.lin(out[predict_type[1]])]
        else:
            out = self.lin(out[predict_type])

        return out

    def encode(self, data, *args, **kwargs):
        x = self.featemb(data.x_dict)
        e = data.edge_index_dict
        return self.forward(x, e)

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf(z)
        return out
