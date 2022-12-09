import sys
import torch
import pytorch_lightning as pl

from trainer import train_model
from utils import parse_args, load_config

if __name__ == "__main__":
    pl.seed_everything(42)
    args = parse_args(sys.argv[1:])
    config = load_config(args.config)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    for arg in vars(args):
        config[arg] = getattr(args, arg)
    print(config)
    model, result = train_model(**config)
