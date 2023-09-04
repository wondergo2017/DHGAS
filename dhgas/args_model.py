import argparse
from dhgas.models import Sta_MODEL, Homo_MODEL
import os


def get_args(args=None):
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument("--dataset", type=str, default="Aminer")
    parser.add_argument("--model", type=str)
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

    args = parser.parse_args(args)

    # post
    args.device = f"cuda:{args.device}"
    args.dynamic = args.model not in Sta_MODEL
    args.homo = args.model in Homo_MODEL
    args.test_full = not args.dynamic  # static model use full training data for testing
    os.makedirs(args.log_dir, exist_ok=True)
    return args
