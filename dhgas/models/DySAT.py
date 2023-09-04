import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
from .dysat.layers import TemporalAttentionLayer

# from dhnas.models.dysat.layers import GATLayer as StructuralAttentionLayer
from .dysat.layers import StructuralAttentionLayer2 as StructuralAttentionLayer
import argparse
from dhgas.data.utils import make_hodata


def get_args(args=""):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_steps",
        type=int,
        nargs="?",
        default=16,
        help="total time steps used for train, eval and test",
    )
    # Experimental settings.
    parser.add_argument(
        "--dataset", type=str, nargs="?", default="Enron", help="dataset name"
    )
    parser.add_argument(
        "--GPU_ID", type=int, nargs="?", default=0, help="GPU_ID (0/1 etc.)"
    )
    parser.add_argument("--epochs", type=int, nargs="?", default=200, help="# epochs")
    parser.add_argument(
        "--val_freq",
        type=int,
        nargs="?",
        default=1,
        help="Validation frequency (in epochs)",
    )
    parser.add_argument(
        "--test_freq",
        type=int,
        nargs="?",
        default=1,
        help="Testing frequency (in epochs)",
    )
    parser.add_argument(
        "--batch_size", type=int, nargs="?", default=512, help="Batch size (# nodes)"
    )
    parser.add_argument(
        "--featureless",
        type=bool,
        nargs="?",
        default=True,
        help="True if one-hot encoding.",
    )
    parser.add_argument("--early_stop", type=int, default=10, help="patient")
    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    # Tunable hyper-params
    parser.add_argument(
        "--residual", type=bool, nargs="?", default=True, help="Use residual"
    )
    # Number of negative samples per positive pair.
    parser.add_argument(
        "--neg_sample_size",
        type=int,
        nargs="?",
        default=10,
        help="# negative samples per positive",
    )
    # Walk length for random walk sampling.
    parser.add_argument(
        "--walk_len",
        type=int,
        nargs="?",
        default=20,
        help="Walk length for random walk sampling",
    )
    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument(
        "--neg_weight",
        type=float,
        nargs="?",
        default=1.0,
        help="Weightage for negative samples",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        nargs="?",
        default=0.01,
        help="Initial learning rate for self-attention model.",
    )
    parser.add_argument(
        "--spatial_drop",
        type=float,
        nargs="?",
        default=0.1,
        help="Spatial (structural) attention Dropout (1 - keep probability).",
    )
    parser.add_argument(
        "--temporal_drop",
        type=float,
        nargs="?",
        default=0.5,
        help="Temporal attention Dropout (1 - keep probability).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        nargs="?",
        default=0.0005,
        help="Initial learning rate for self-attention model.",
    )
    # Architecture params
    parser.add_argument(
        "--structural_head_config",
        type=str,
        nargs="?",
        default="16",  # 16,8,8
        help="Encoder layer config: # attention heads in each GAT layer",
    )
    parser.add_argument(
        "--structural_layer_config",
        type=str,
        nargs="?",
        default="64",
        help="Encoder layer config: # units in each GAT layer",
    )
    parser.add_argument(
        "--temporal_head_config",
        type=str,
        nargs="?",
        default="16",  # 16
        help="Encoder layer config: # attention heads in each Temporal layer",
    )
    parser.add_argument(
        "--temporal_layer_config",
        type=str,
        nargs="?",
        default="128",
        help="Encoder layer config: # units in each Temporal layer",
    )
    parser.add_argument(
        "--position_ffn",
        type=str,
        nargs="?",
        default="True",
        help="Position wise feedforward",
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs="?",
        default=-1,
        help="Window for temporal attention (default : -1 => full)",
    )
    args = parser.parse_args(args)
    return args


class DySAT(nn.Module):
    def __init__(
        self,
        hid_dim,
        time_length,
        num_layers,
        n_heads,
        metadata,
        predict_type,
        featemb=None,
        nclf_linear=None,
    ):
        super(DySAT, self).__init__()
        args = get_args()
        self.args = args
        if args.window < 0:
            self.num_time_steps = time_length
        else:
            self.num_time_steps = min(
                time_length, args.window + 1
            )  # window = 0 => only self.
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.n_heads = n_heads

        # self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        # self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        # self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        # self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop

        self.structural_attn, self.temporal_attn = self.build_model()

        self.bceloss = BCEWithLogitsLoss()
        # self.hlinear = HLinear(hid_dim, metadata, act='None')
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x
        self.nclf = nclf_linear

    def encode(self, graphs, *args, **kwargs):
        # Structural Attention forward
        structural_out = []
        for t in range(0, self.num_time_steps):
            data = graphs[t]
            x_dict = self.featemb(data.x_dict)
            # x_dict = self.hlinear(x_dict)
            e_dict = data.edge_index_dict
            x, e, mask, data = make_hodata(x_dict, e_dict, self.predict_type)
            for i in range(len(self.structural_attn)):
                x = self.structural_attn[i](x, data)

            structural_out.append(x)
        structural_outputs = [
            x[:, None, :] for x in structural_out
        ]  # list of [Ni, 1, F]

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_dim).to(
                out.device
            )
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(
            structural_outputs_padded, dim=1
        )  # [N, T, F]

        # Temporal Attention forward
        # temporal_out = structural_outputs_padded
        temporal_out = self.temporal_attn(structural_outputs_padded)

        x = temporal_out[:, -1, :]

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

    def build_model(self):
        hid_dim = self.hid_dim

        # 1: Structural Attention Layers
        structural_attention_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = StructuralAttentionLayer(
                input_dim=hid_dim,
                output_dim=hid_dim,
                n_heads=self.n_heads,
                attn_drop=self.spatial_drop,
                ffd_drop=self.spatial_drop,
                residual=self.args.residual,
            )
            # layer= GATLayer(input_dim,self.structural_layer_config[i],self.structural_head_config[i],dropout=self.spatial_drop)
            structural_attention_layers.append(layer)

        # 2: Temporal Attention Layers
        temporal_attention_layers = nn.Sequential()
        for i in range(1):
            layer = TemporalAttentionLayer(
                input_dim=hid_dim,
                n_heads=self.n_heads,
                num_time_steps=self.num_time_steps,
                attn_drop=self.temporal_drop,
                residual=self.args.residual,
            )
            temporal_attention_layers.add_module(
                name="temporal_layer_{}".format(i), module=layer
            )

        return structural_attention_layers, temporal_attention_layers

    def get_loss(self, feed_dict):
        node_1, node_2, node_2_negative, graphs = feed_dict.values()
        # run gnn
        final_emb = self.forward(graphs)  # [N, T, F]
        self.graph_loss = 0
        for t in range(self.num_time_steps - 1):
            emb_t = final_emb[:, t, :].squeeze()  # [N, F]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]
            pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(
                source_node_emb[:, None, :] * tart_node_neg_emb, dim=2
            ).flatten()
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.ones_like(neg_score))
            graphloss = pos_loss + self.args.neg_weight * neg_loss
            self.graph_loss += graphloss
        return self.graph_loss
