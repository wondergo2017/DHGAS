import json
import torch
import os.path as osp

import numpy as np
from dhgas.utils import EarlyStopping
from tqdm import tqdm
import time
from torch import nn
from torch.nn import functional as F
from dhgas.utils import setup_seed


def train(
    model, optimizer, criterion, train_data, culmulate=1, grad_clip=0, device="cpu"
):
    model.train()

    losses = []
    for support, query in train_data:
        z = model.encode(support)
        out = model.decode_nclf(z)
        loss = criterion(out, query.x)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


@torch.no_grad()
def test(model, data, device="cpu"):
    def test_one(model, data):
        support, query = data
        model.eval()
        z = model.encode(support)
        out = model.decode_nclf(z)
        mae = F.l1_loss(out, query.x).item()
        return mae

    if isinstance(data, list):
        aucs = [test_one(model, d) for d in data]
        return np.mean(aucs)
    return test_one(model, data)


def train_till_end(
    model,
    optimizer,
    criterion,
    dataset,
    args,
    max_epochs,
    patience,
    disable_progress=False,
    writer=None,
    grad_clip=0,
    device="cpu",
):
    # procedure
    setup_seed(args.seed)
    start_time = time.time()
    best_val_auc = final_test_auc = 1e8
    earlystop = EarlyStopping(mode="min", patience=patience)
    with tqdm(range(max_epochs), disable=disable_progress) as bar:
        for epoch in bar:
            loss = train(
                model,
                optimizer,
                criterion,
                dataset.train_dataset,
                grad_clip=grad_clip,
                device=device,
            )
            # train_auc = test(model, dataset.train_dataset,device=device)
            train_auc = loss
            val_auc = test(model, dataset.val_dataset, device=device)
            ts = time.time()
            test_auc = test(model, dataset.test_dataset, device=device)
            print("test ", (time.time() - ts) / len(dataset.test_dataset))
            if val_auc < best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc
            bar.set_postfix(
                loss=loss,
                train_auc=train_auc,
                val_auc=val_auc,
                test_auc=test_auc,
                btest_auc=final_test_auc,
            )
            if writer:
                writer.add_scalar("Model/train_loss", loss, epoch)
                writer.add_scalar("Model/val_auc", val_auc, epoch)
                writer.add_scalar("Model/test_auc", test_auc, epoch)

            if earlystop.step(val_auc):
                break

    return {
        "test_auc": final_test_auc,
        "val_auc": best_val_auc,
        "train_auc": train_auc,
        "epoch": epoch,
        "time": time.time() - start_time,
        "time_per_epoch": (time.time() - start_time) / (epoch + 1),
    }


class NodePredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int):
        """

        :param n_inp      : int, input dimension
        :param n_classes  : int, number of classes
        """
        super().__init__()

        self.fc1 = nn.Linear(n_inp, n_inp)
        self.fc2 = nn.Linear(n_inp, n_classes)

    def forward(self, node_feat: torch.tensor):
        """

        :param node_feat: torch.tensor
        """

        node_feat = F.relu(self.fc1(node_feat))
        pred = F.relu(self.fc2(node_feat))

        return pred
