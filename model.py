# Standard libraries
import numpy as np
import math

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


def sdp_attention(q, k, v, mask=None, padding_mask=None):
    """Performs scaled dot product attention"""
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
    if padding_mask is not None:
        padding_mask = torch.unsqueeze(padding_mask, 1)
        attn_logits = attn_logits.masked_fill(padding_mask == 1, float('-inf'))
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 1, float('-inf'))
    attention = F.softmax(attn_logits, dim=-1)
    attention = torch.nan_to_num(attention)
    values = torch.matmul(attention, v)
    return values, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, model_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.ModuleList([nn.Linear(input_dim, model_dim) for _ in range(3)])
        self.o_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, mask=None, padding_mask=None, return_attention=False):
        # Query is of shape (B, L, E)
        q_batch_size, q_seq_length, q_embed_dim = query.size()
        k_batch_size, k_seq_length, k_embed_dim = key.size()
        v_batch_size, v_seq_length, v_embed_dim = value.size()

        query, key, value = [layer(x) for layer, x in zip(self.qkv_proj, (query, key, value))]
        query = query.reshape(q_batch_size, self.num_heads, q_seq_length, self.head_dim)
        key = key.reshape(k_batch_size, self.num_heads, k_seq_length, self.head_dim)
        value = value.reshape(k_batch_size, self.num_heads, v_seq_length, self.head_dim)
        # Determine value outputs
        values, attention = sdp_attention(query, key, value, mask=mask, padding_mask=padding_mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(q_batch_size, q_seq_length, q_embed_dim)
        output = self.o_proj(values)

        if return_attention:
            return output, attention
        else:
            return output


class EncoderBlock(nn.Module):

    def __init__(self, seq_embed_dim, model_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(seq_embed_dim, model_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, model_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, padding_mask=None):
        # Attention part
        attn_out = self.self_attn(x, x, x, mask=mask, padding_mask=padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, seq_embed_dim, model_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(seq_embed_dim, model_dim, num_heads)
        self.src_attn = MultiheadAttention(model_dim, model_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, model_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, tgt_padding_mask, cross_padding_mask):
        m = memory

        attn_out = self.self_attn(x, x, x, mask=tgt_mask, padding_mask=tgt_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        attn_out = self.src_attn(x, m, m, mask=None, padding_mask=cross_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)

        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm3(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None, padding_mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask, padding_mask=padding_mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, memory, tgt_mask, tgt_padding_mask, cross_padding_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_padding_mask, cross_padding_mask)
        return x


class TransformerModel(nn.Module):
    """
    Transformer Module
    """
    def __init__(self, src_vocab, tgt_vocab, embed_dim, model_dim, num_layers=6, num_heads=8):
        super().__init__()

        self.positional_encoding = PositionalEncoding(model_dim)

        self.transformer_encoder = TransformerEncoder(num_layers=num_layers,
                                                      seq_embed_dim=embed_dim,
                                                      model_dim=model_dim,
                                                      num_heads=num_heads,
                                                      dim_feedforward=2048,
                                                      dropout=0.1)

        self.transformer_decoder = TransformerDecoder(num_layers=num_layers,
                                                      seq_embed_dim=embed_dim,
                                                      model_dim=model_dim,
                                                      num_heads=num_heads,
                                                      dim_feedforward=2048,
                                                      dropout=0.1)

        self.src_embed = nn.Embedding(src_vocab, embed_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab, embed_dim)

        self.generator = nn.Linear(model_dim, tgt_vocab)

        self.d_model = model_dim

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, cross_padding_mask):

        src = self.src_embed(src)
        src = self.positional_encoding(src)
        encoded_output = self.transformer_encoder(src, src_mask, src_padding_mask)

        tgt = self.tgt_embed(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.transformer_decoder(tgt, encoded_output, tgt_mask, tgt_padding_mask, cross_padding_mask)

        output = self.generator(output)

        return output


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
