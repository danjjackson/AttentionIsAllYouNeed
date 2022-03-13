import torch
import spacy

from torchtext.data import get_tokenizer
from torchtext.datasets import Multi30k

from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def build_vocab(token_transforms):

    vocab_transform = {}
    special_symbols = ['<pad>', '<unk>' '<bos>', '<eos>']

    # helper function to yield list of tokens
    def yield_tokens(data_iter, ln: str):
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield token_transforms[ln](data_sample[language_index[ln]])

    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        vocab_transform[language] = build_vocab_from_iterator(yield_tokens(train_iter, language),
                                                              min_freq=1,
                                                              specials=special_symbols,
                                                              special_first=True)
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    return vocab_transform


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


def create_data_loaders(batch_size=128):

    token_transforms = dict()

    # token_transforms[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transforms[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

    vocab_transforms = build_vocab(token_transforms)
    SRC_VOCAB_SIZE = len(vocab_transforms[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transforms[TGT_LANGUAGE])

    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[language] = sequential_transforms(token_transforms[language],  # Tokenisation
                                                         vocab_transforms[language],  # Numericalisation
                                                         tensor_transform)  # Add BOS/EOS and create tensor

    # function to collate data samples into batch tensors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    train_iter, val_iter, test_iter = Multi30k()
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    valid_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    test_dataloader = DataLoader(test_iter, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)

    return train_dataloader, valid_dataloader, test_dataloader, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
