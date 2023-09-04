from dhgas.models import DHSearcher
import torch
from copy import deepcopy
from dhgas.args_search import get_args
from dhgas.data import load_data
from dhgas.models.load_model import load_pre_post, load_lazy_hetero_weights
from dhgas.models.DHSpaceSearch import DHNet, DHSpace, DHSearcher
from dhgas.trainer import load_trainer
from dhgas.utils import setup_seed
import json
import os

if __name__ == "__main__":
    args = get_args()

    # dataset
    dataset, args = load_data(args)
    hid_dim, metadata, twin, KTO, KN, KR, n_heads, predict_type, device = (
        args.hid_dim,
        dataset.metadata,
        args.twin,
        args.KTO,
        args.KN,
        args.KR,
        args.n_heads,
        args.predict_type,
        args.device,
    )

    # model
    setup_seed(args.seed)
    featemb, nclf_linear = load_pre_post(args, dataset)
    Net = DHNet
    n_layers = 2  # reuse the searched first layer for efficiency
    dhspaces = []
    for i in range(n_layers - 1):
        dhspaces.append(
            DHSpace(
                hid_dim,
                metadata,
                twin,
                KTO,
                KN,
                KR,
                n_heads,
                causal_mask=args.causal_mask,
                last_mask=False,
                full_mask=True,
                rel_time_type=args.rel_time_type,
                time_patch_num=args.patch_num,
                norm=args.norm,
                hupdate=args.hupdate,
            )
        )
    dhspaces.append(
        DHSpace(
            hid_dim,
            metadata,
            twin,
            KTO,
            KN,
            KR,
            n_heads,
            causal_mask=args.causal_mask,
            last_mask=True,
            full_mask=True,
            rel_time_type="relative",
            time_patch_num=1,
            norm=args.norm,
            hupdate=args.hupdate,
        )
    )
    model = Net(
        hid_dim,
        twin,
        metadata,
        dhspaces,
        predict_type,
        featemb=featemb,
        nclf_linear=nclf_linear,
        hlinear_act=args.hlinear_act,
    )
    load_lazy_hetero_weights(args, dataset, model)

    # device
    model = model.to(device)
    dataset.to(device)
    if args.resume:
        model.load_state_dict(
            torch.load(os.path.join(args.supernet_dir, f"checkpoint{args.resume}"))
        )

    # trainer
    trainer, criterion = load_trainer(args)
    searcher = DHSearcher(
        criterion, args.supernet_dir, None, n_warmup=args.n_warmup, args=args
    )
    best_pop = searcher.search(model, dhspaces, dataset, topk=args.topk)

    # logs
    info_dict = args.__dict__
    configs = []
    for i, (config, estimation) in enumerate(best_pop):
        arch_info_dict = deepcopy(info_dict)
        arch_dir = os.path.join(args.arch_dir, f"{i}")
        os.makedirs(arch_dir, exist_ok=True)
        fconfig = os.path.join(arch_dir, "config")
        torch.save(config, fconfig)
        json.dump(
            info_dict,
            open(os.path.join(arch_dir, "supernet.json"), "w"),
            indent=4,
            sort_keys=True,
        )
        open(os.path.join(arch_dir, "config_read.txt"), "w").write(f"{config}")
        configs.append(arch_dir)

    # rerun the searched model
    torch.cuda.empty_cache()
    dev = args.device.split(":")[-1]
    arch_dir = configs[0]
    results = []
    for seed in [11, 22, 33]:
        log_dir = f"{arch_dir}/{seed}"
        os.makedirs(log_dir, exist_ok=True)
        cmd = f'python run_model.py --seed {seed} --device {dev} --model DHSpace --twin {args.twin} --log_dir "{log_dir}" --dhconfig "{arch_dir}" --dataset {args.dataset} --n_heads {args.n_heads} --norm {args.norm} --hlinear_act {args.hlinear_act} --lr {args.lr} --wd {args.wd} > "{log_dir}/log.txt"'
        with open(os.path.join(log_dir, "cmd.sh"), "w") as f:
            f.write(cmd)
        os.system(cmd)
        results.append(json.load(open(os.path.join(log_dir, "info.json")))["test_auc"])
    with open(os.path.join(arch_dir, "results.txt"), "w") as f:
        f.write(str(results))
