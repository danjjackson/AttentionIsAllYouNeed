import torch

PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ['<pad>', '<unk>' '<bos>', '<eos>']

CHECKPOINT_PATH = 'saved_models'