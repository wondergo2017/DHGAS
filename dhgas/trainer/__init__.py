import json
import os.path as osp
import torch
import torch.nn.functional as F


def load_trainer(args):
    if args.dataset in "Aminer Ecomm".split():
        from .lpred import train_till_end as trainer
    elif args.dataset in "Yelp-nc".split():
        from .nclf import train_till_end as trainer
    elif args.dataset in "covid ".split():
        from .nreg import train_till_end as trainer

    if args.dataset in "Aminer Ecomm".split():
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.dataset in "Yelp-nc".split():
        criterion = F.cross_entropy
    elif args.dataset in "covid ".split():
        criterion = F.l1_loss
    return trainer, criterion


def load_train_test(args):
    if args.dataset in "Aminer Ecomm".split():
        from .lpred import train, test
    elif args.dataset in "Yelp-nc".split():
        from .nclf import train, test
    elif args.dataset in "covid ".split():
        from .nreg import train, test
    return train, test


def log_train(log_dir, args, train_dict, writer, **kwargs):

    info_dict = args.__dict__
    measure_dict = {}
    for name in "test_auc val_auc train_auc".split():
        measure_dict[name] = train_dict[name]
        del train_dict[name]

    info_dict.update(train_dict)
    info_dict.update(kwargs)
    if writer:
        writer.add_hparams(info_dict, measure_dict)

    info_dict.update(measure_dict)

    json.dump(
        info_dict, open(osp.join(log_dir, "info.json"), "w"), indent=4, sort_keys=True
    )
