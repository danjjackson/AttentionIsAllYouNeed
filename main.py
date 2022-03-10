import argparse
import sys

import pytorch_lightning as pl

from trainer import train_model

def select_gpu(gpu_id: (int, str, list)):
    if isinstance(gpu_id, (list, tuple)):
        gpu_id = ",".join([str[i] for i in gpu_id])
    else:
        gpu_id = str(gpu_id)

    return gpu_id


def parse_args(cmdline):
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('--gpu', default=0, type=int, metavar='N', required=True,
                          help='Index of the GPU to use. Use -1 to use CPU only')
    optional.add_argument("--batch_size", default=64, type=int, required=False)
    _args = parser.parse_args(cmdline)
    return _args


if __name__ == "__main__":
    pl.seed_everything(42)
    args = parse_args(sys.argv[1:])
    model, result = train_model(args)
