from .GCN import GCN
from .GAT import GAT
from .RGCN import RGCN
from .HGT import HGT
from .DyHATR import DyHATR
from .HTGNN import HTGNN
from .DHSpace import DHSpace, DHNet
from .HLinear import FeatEmbed
from torch import nn
import torch
import json
import os
from ..utils import count_parameters, cnt2str

from ..utils import setup_seed


def load_pre_post(args, dataset):
    feat_hid_dim = args.hid_dim
    if args.dataset == "Aminer":
        feat_hid_dim = 32 if args.homo else args.hid_dim
        featemb = FeatEmbed(dataset.dataset, "author venue".split(), feat_hid_dim)
        nclf_linear = None
    elif args.dataset == "Ecomm":
        featemb = FeatEmbed(dataset.dataset, "user item".split(), feat_hid_dim)
        nclf_linear = None
    elif args.dataset == "Yelp-nc":
        featemb = None
        nclf_linear = nn.Linear(args.hid_dim, args.num_classes)
    elif args.dataset == "covid":
        from dhgas.trainer.nreg import NodePredictor

        featemb = None
        nclf_linear = NodePredictor(n_inp=8, n_classes=1)
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")
    return featemb, nclf_linear


def load_backbone(args, dataset, featemb, nclf_linear):
    in_dim, hid_dim, out_dim = args.in_dim, args.hid_dim, args.out_dim
    n_layers, metadata, predict_type, n_heads, time_window, device, norm = (
        args.n_layers,
        dataset.metadata,
        args.predict_type,
        args.n_heads,
        args.twin,
        args.device,
        args.norm,
    )
    dhconfig = args.dhconfig
    model = args.model
    if model == "GCN":
        from .GCN import GCN as Net

        model = Net(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
    elif model == "GAT":
        from .GAT import GAT as Net

        model = Net(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            heads=n_heads,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
    elif model == "RGCN":
        from .RGCN import RGCN as Net

        model = Net(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
    elif model == "HGT":
        from .HGT import HGT as Net

        model = Net(
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
    elif model == "HGT+":
        from .HGT import HGT as Net

        model = Net(
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            use_RTE=True,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
    elif model == "DyHATR":
        from .DyHATR import DyHATR as Net

        model = Net(
            in_dim,
            hid_dim,
            n_layers=1,
            n_heads=n_heads,
            time_window=time_window,
            metadata=metadata,
            predict_type=predict_type,
            dropout=0.2,
            edge_layers=n_layers,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )  # use edge_layers=2,n_heads=4,dropout=0.2 in original code.
    elif model == "HTGNN":
        from .HTGNN import HTGNN as Net

        model = Net(
            hid_dim,
            hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            time_window=time_window,
            norm=False,
            metadata=metadata,
            device=device,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
        # use n_heads=1,n_layers=2,norm=True in original code. but layernorm can cause very low performance. This may be due to embedding layer.
    elif "DHSpace" in model:
        from .DHSpace import DHNet as Net
        from .DHSpace import DHSpace

        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                if model == "DHSpace":
                    dhspaces.append(
                        DHSpace(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                        ).assign_basic_arch(["causal", "node_hetero"])
                    )
                elif model == "DHSpaceS":
                    dhspaces.append(
                        DHSpace(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                        ).assign_basic_arch(["causal"])
                    )
                elif model == "DHSpaceF":
                    dhspaces.append(
                        DHSpace(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                        ).assign_basic_arch(["full", "node_hetero"])
                    )
                elif model == "DHSpaceH":
                    dhspaces.append(
                        DHSpace(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                        ).assign_basic_arch(["full", "node_hetero", "rel_hetero"])
                    )

            if model == "DHSpace":
                dhspaces.append(
                    DHSpace(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["last", "node_hetero"])
                )
            elif model == "DHSpaceS":
                dhspaces.append(
                    DHSpace(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["last"])
                )
            elif model == "DHSpaceF":
                dhspaces.append(
                    DHSpace(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["last", "node_hetero"])
                )
            elif model == "DHSpaceH":
                dhspaces.append(
                    DHSpace(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["last", "node_hetero", "rel_hetero"])
                )
        else:
            cfg = torch.load(
                os.path.join(dhconfig, "config")
            )  # config only determines Ato,AN,AR
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpace(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    rel_time_type=info["rel_time_type"],
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                )
                dhspace.assign_arch(a)
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    else:
        raise NotImplementedError(f"Unexpected model {model}")
    return model


def load_lazy_hetero_weights(args, dataset, model):
    with torch.no_grad():  # Initialize lazy modules.
        if args.dataset in "Aminer Ecomm".split():
            out = model.encode(dataset.val_dataset[0])
        elif args.dataset in "Yelp-nc".split():
            out = model.encode(dataset.val_dataset[0][0])
        elif args.dataset in "covid ".split():
            out = model.encode(dataset.val_dataset[0][0])


def load_model(args, dataset):
    setup_seed(args.seed)
    featemb, nclf_linear = load_pre_post(args, dataset)
    model = load_backbone(args, dataset, featemb, nclf_linear)
    load_lazy_hetero_weights(args, dataset, model)
    return model
