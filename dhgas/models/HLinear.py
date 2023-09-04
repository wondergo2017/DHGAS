import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import Linear
from torch.nn import LayerNorm


class HLinear(nn.Module):
    def __init__(self, out_dim, metadata, act="tanh"):
        super(HLinear, self).__init__()
        self.out_dim = out_dim
        node_types = metadata[0]
        self.adapt_ws = nn.ModuleDict()
        for nt in node_types:
            self.adapt_ws[nt] = Linear(-1, out_dim)
        if act == "tanh":
            self.act = torch.tanh
        elif act == "relu":
            self.act = F.relu
        elif act == "None":
            self.act = lambda x: x
        else:
            raise NotImplementedError(f"Unknown HLinear activation {act}")

    def __getitem__(self, index):
        return self.adapt_ws[index]

    def reset_parameters(self):
        for k, lin in self.adapt_ws.items():
            lin.reset_parameters()

    def forward(self, x_dict, *args, **kwargs):
        y_dict = {}
        for nt in x_dict:
            y_dict[nt] = self.act(self.adapt_ws[nt](x_dict[nt]))
        return y_dict


class FeatEmbed(nn.Module):
    def __init__(self, dataset, emb_types, embed_dim):
        super(FeatEmbed, self).__init__()
        embeds = nn.ModuleDict()
        for tp in emb_types:
            embeds[tp] = torch.nn.Embedding(dataset[tp].x.shape[0], embed_dim)
        self.embeds = embeds

    def reset_parameters(self):
        for tp, emb in self.embeds.items():
            emb.reset_parameters()

    def forward(self, x_dict):
        y_dict = {}
        for tp in x_dict:
            if tp in self.embeds:
                y_dict[tp] = self.embeds[tp](x_dict[tp])
            else:
                y_dict[tp] = x_dict[tp]
        return y_dict


class HLayerNorm(nn.Module):
    def __init__(self, out_dim, metadata):
        super(HLayerNorm, self).__init__()
        self.out_dim = out_dim
        node_types = metadata[0]
        self.hfuncs = nn.ModuleDict()
        for nt in node_types:
            self.hfuncs[nt] = LayerNorm(out_dim)

    def __getitem__(self, index):
        return self.hfuncs[index]

    def reset_parameters(self):
        for k, func in self.hfuncs.items():
            func.reset_parameters()

    def forward(self, x_dict, *args, **kwargs):
        y_dict = {}
        for nt in x_dict:
            y_dict[nt] = self.hfuncs[nt](x_dict[nt])
        return y_dict
