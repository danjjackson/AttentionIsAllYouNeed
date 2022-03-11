import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os

from model import TransformerModel, CosineWarmupScheduler
from dataset import create_data_loaders

PAD_IDX = 0
CHECKPOINT_PATH = 'saved_models'
save_name = 'test'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Transformer(pl.LightningModule):

    def __init__(self, n_src, n_tgt, embed_dim, model_dim, num_heads, num_layers, lr, warmup, max_iters, PAD_IDX=0):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._make_model()
        self.loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        # self.acc = pl.metrics.Accuracy()

    def _make_model(self):
        self.model = TransformerModel(self.hparams.n_src,
                                      self.hparams.n_tgt,
                                      self.hparams.embed_dim,
                                      self.hparams.model_dim)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, x, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, cross_padding_mask=None):
        x = self.model(x, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, cross_padding_mask)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _calculate_loss(self, batch):
        src, tgt = batch
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, cross_padding_mask = create_mask(src, tgt)
        output = self.forward(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, cross_padding_mask)
        loss = self.loss(output.transpose(1, 2), tgt)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log(f'train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log(f'val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log(f'test_loss', loss)


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_mask = tgt_mask.type_as(tgt)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    src_mask = src_mask.type_as(src)

    src_padding_masks = []
    tgt_padding_masks = []
    cross_padding_masks = []

    src_padding = (src == PAD_IDX)
    tgt_padding = (tgt == PAD_IDX)

    def make_mask(x, y):
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        mask = torch.logical_or(x.transpose(0, 1), y)
        return mask

    for src_row, tgt_row in zip(src_padding, tgt_padding):
        src_tmp_mask = make_mask(src_row, src_row)
        tgt_tmp_mask = make_mask(tgt_row, tgt_row)
        cross_tmp_mask = make_mask(tgt_row, src_row)

        src_padding_masks.append(src_tmp_mask)
        tgt_padding_masks.append(tgt_tmp_mask)
        cross_padding_masks.append(cross_tmp_mask)

    src_padding_mask = torch.stack(src_padding_masks)
    src_padding_mask = src_padding_mask.type_as(src)
    tgt_padding_mask = torch.stack(tgt_padding_masks)
    tgt_padding_mask = tgt_padding_mask.type_as(tgt)
    cross_padding_mask = torch.stack(cross_padding_masks)
    cross_padding_mask = cross_padding_mask.type_as(src)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, cross_padding_mask


def train_model(args=None):

    train_loader, val_loader, test_loader, n_src, n_tgt = create_data_loaders(batch_size=args.batch_size)

    # Create a PyTorch Lightning trainer with the generation callback

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                          # Where to save models
                         gpus=1 if str(device) == "cuda:0" else 0,                                           # We run on a single GPU (if possible)
                         max_epochs=180,                                                                     # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_loss"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch")],                                           # Log learning rate every epoch
                         progress_bar_refresh_rate=1)
    model = Transformer(embed_dim=512,
                        model_dim=512,
                        n_src=n_src,
                        n_tgt=n_tgt,
                        num_heads=8,
                        num_layers=6,
                        lr=5e-4,
                        warmup=100,
                        max_iters=trainer.max_epochs+10)

    trainer.fit(model, train_loader, val_loader)

    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["val_acc"]}

    return model, result



