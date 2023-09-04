import argparse
from dhgas.models import Sta_MODEL, Homo_MODEL
import os


def setargs(args, hp):
    for k, v in hp.items():
        setattr(args, k, v)


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cfg", type=int, default=1)
    # basic
    parser.add_argument("--dataset", type=str, default="Aminer")
    parser.add_argument("--model", type=str, default="DHSpace")
    parser.add_argument("--dhconfig", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="logs/tmp")
    parser.add_argument("--device", default="6")
    parser.add_argument("--seed", type=int, default=22)

    # auto
    parser.add_argument("--dynamic", type=int, default=-1)
    parser.add_argument("--homo", type=int, default=-1)
    parser.add_argument("--twin", type=int, default=-1)
    parser.add_argument("--test_full", type=int, default=-1)
    parser.add_argument("--predict_type", type=str, default="")
    parser.add_argument("--in_dim", type=int, default=-1)
    parser.add_argument("--hid_dim", type=int, default=-1)
    parser.add_argument("--out_dim", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=-1)

    # optim
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--cul", type=int, default=1)

    # hp
    parser.add_argument("--norm", type=int, default=1)
    parser.add_argument("--hlinear_act", type=str, default="tanh")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--grad_clip", type=float, default=0)

    # search
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--supernet_early_stop", type=int, default=1000)
    parser.add_argument("--causal_mask", type=int, default=1, help="1 True else False")
    parser.add_argument("--node_entangle_type", type=str, default="None")
    parser.add_argument("--rel_entangle_type", type=str, default="None")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--rel_time_type", type=str, default="relative")
    parser.add_argument("--hupdate", type=int, default=1)
    parser.add_argument("--reset_type", type=int, default=0)
    parser.add_argument("--reset_type2", type=int, default=0)
    parser.add_argument("--patch_num", type=int, default=1)
    parser.add_argument("--KN", type=int, default=2)
    parser.add_argument("--KR", type=int, default=2)
    parser.add_argument("--KTO", type=int, default=10)
    parser.add_argument("--n_warmup", type=int, default=40)
    parser.add_argument("--arch_dir", type=str, default="")
    parser.add_argument("--supernet_dir", type=str, default="")
    args = parser.parse_args(args)

    # full cfg
    if args.use_cfg:
        if args.dataset == "Aminer":
            hp = {
                "patch_num": 2,
                "KN": 5,
                "KR": 4,
                "KTO": 500,
                "n_layers": 3,
                "n_heads": 4,
                "n_warmup": 30,
                "twin": 8,
            }
        elif args.dataset == "Ecomm":
            hp = {
                "patch_num": 2,
                "KN": 5,
                "KR": 3,
                "KTO": 500,
                "n_layers": 2,
                "n_heads": 2,
                "n_warmup": 15,
                "twin": 7,
            }
        elif args.dataset == "Yelp-nc":
            hp = {
                "patch_num": 2,
                "KN": 5,
                "KR": 5,
                "KTO": 500,
                "n_layers": 2,
                "n_heads": 2,
                "n_warmup": 20,
                "twin": 12,
            }
        else:
            raise NotImplementedError(f"dataset {args.dataset} not implemented")
        setargs(args, hp)

    # post
    assert args.model == "DHSpace", "DHSearcher only supports DHSpace"
    args.device = f"cuda:{args.device}"
    args.dynamic = args.model not in Sta_MODEL
    args.homo = args.model in Homo_MODEL
    args.test_full = not args.dynamic  # static model use full training data for testing
    os.makedirs(args.log_dir, exist_ok=True)
    args.supernet_dir = os.path.join(args.log_dir, "supernet/")
    args.arch_dir = os.path.join(args.log_dir, "archs/")
    for d in [args.log_dir, args.supernet_dir, args.arch_dir]:
        os.makedirs(d, exist_ok=True)
    return args
