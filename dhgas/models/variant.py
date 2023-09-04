import torch
from torch import nn
from dhgas.models.dysat.layers import TemporalAttentionLayer


class ModelAddT(nn.Module):
    def __init__(self, model, time_window, Tattn_drop, Tattn_residual):
        super(ModelAddT, self).__init__()
        self.model = model
        self.TAttn = TemporalAttentionLayer(
            64, 8, time_window, attn_drop=Tattn_drop, residual=Tattn_residual
        )

    def encode(self, graphs, *args, **kwargs):
        xlist = []
        for g in graphs:
            x = self.model.encode(g)
            xlist.append(x)
        xs = torch.stack(xlist, dim=1)
        output = self.TAttn(xs)
        return output[:, -1, :]

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)


from dhgas.data.crossdomain import time_merge_homo as time_merge


class SlowFast(nn.Module):
    def __init__(self, model, time_window, Tattn_drop, Tattn_residual, fuse="add"):
        super(SlowFast, self).__init__()
        self.model = model
        self.TAttn = TemporalAttentionLayer(
            64, 8, time_window, attn_drop=Tattn_drop, residual=Tattn_residual
        )
        self.fuse = fuse
        self.linear = nn.Linear(128, 64)

    def encode(self, graphs, *args, **kwargs):
        graph = time_merge(graphs)
        xslow = self.model.encode(graph)

        xlist = []
        for g in graphs:
            x = self.model.encode(g)
            xlist.append(x)
        xs = torch.stack(xlist, dim=1)
        xfast = self.TAttn(xs)[:, -1, :]

        if self.fuse == "add":
            output = xfast + xslow
        elif self.fuse == "concat":
            output = torch.cat([xslow, xfast], dim=-1)
            output = self.linear(output)
        return output

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)
