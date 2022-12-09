import torch
import spacy

from torchtext.data import get_tokenizer
from torchtext.datasets import Multi30k

from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from const import SPECIALS, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX

class Multi30kDataset(Dataset):
    def __init__(self, mode, src_language, tgt_language):
        self.dataset_list = list(Multi30k(
            split=mode,
            language_pair=(
                src_language,
                tgt_language
            )
        ))

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        src, tgt = self.dataset_list[idx]
        return src, tgt


class Vocab:
    def __init__(self, src_language, tgt_language):
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.token_transforms = self.get_token_transforms()
        self.vocab_transforms = self.build_vocab()
        self.src_vocab_size = len(self.vocab_transforms[src_language])
        self.tgt_vocab_size = len(self.vocab_transforms[tgt_language])
        self.text_transform = self.build_text_transform()

    def get_token_transforms(self):

        token_transforms = {
            'de': get_tokenizer('spacy', language='de_core_news_sm'),
            'en': get_tokenizer('spacy', language='en_core_web_sm')
        }
        return token_transforms

    def build_vocab(self):

        vocab_transform = {}
        language_index = {self.src_language: 0, self.tgt_language: 1}

        # helper function to yield list of tokens
        def yield_tokens(data_iter, tokeniser, language_idx):

            for data_sample in data_iter:
                yield tokeniser(data_sample[language_idx])

        for language in [self.src_language, self.tgt_language]:
            train_iter = Multi30k(
                split='train',
                language_pair=(
                    self.src_language,
                    self.tgt_language
                )
            )

            vocab_transform[language] = build_vocab_from_iterator(
                yield_tokens(
                    train_iter,
                    self.token_transforms[language],
                    language_index[language]),
                min_freq=1,
                specials=SPECIALS,
                special_first=True)

            vocab_transform[language].set_default_index(UNK_IDX)

        return vocab_transform

    @staticmethod
    def tensor_transform(token_ids):
        return torch.cat(
            (torch.tensor([BOS_IDX]),
             torch.tensor(token_ids),
             torch.tensor([EOS_IDX]))
        )

    @staticmethod
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    def build_text_transform(self):
        text_transform = {}
        for language in [self.src_language, self.tgt_language]:
            text_transform[language] = self.sequential_transforms(
                self.token_transforms[language],  # Tokenisation
                self.vocab_transforms[language],  # Numericalisation
                self.tensor_transform
            )
        return text_transform

    def collate_fn(self, batch, batch_first=True):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(
                self.text_transform[self.src_language](src_sample.rstrip("\n"))
            )
            tgt_batch.append(
                self.text_transform[self.tgt_language](tgt_sample.rstrip("\n"))
            )

        src_batch = pad_sequence(
            src_batch,
            batch_first=batch_first,
            padding_value=PAD_IDX
        )
        tgt_batch = pad_sequence(
            tgt_batch,
            batch_first=batch_first,
            padding_value=PAD_IDX
        )
        return src_batch, tgt_batch
    def create_data_loaders(self, batch_size):

        # function to collate data samples into batch tensor
        data_loaders = {
            dataset: DataLoader(
                Multi30kDataset(
                    dataset,
                    self.src_language,
                    self.tgt_language
                ),
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                num_workers=16
            ) for dataset in ['train', 'valid', 'test']
        }

        return data_loaders