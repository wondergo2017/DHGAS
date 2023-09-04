import torch
from dhgas.data import load_data
from dhgas.models import load_model
from dhgas.args_model import get_args
from dhgas.trainer import load_trainer

# args
args = get_args()

# dataset
dataset, args = load_data(args)

# model
model = load_model(args, dataset)

# device
model = model.to(args.device)
dataset.to(args.device)

# train
trainer, criterion = load_trainer(args)
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=args.lr, weight_decay=args.wd
)
train_dict = trainer(
    model,
    optimizer,
    criterion,
    dataset,
    args,
    args.max_epochs,
    args.patience,
    disable_progress=False,
    writer=None,
    grad_clip=args.grad_clip,
    device=args.device,
)
print(f"Final Test: {train_dict['test_auc']:.4f}")

# close
from dhgas.trainer import log_train

log_train(args.log_dir, args, train_dict, None)
