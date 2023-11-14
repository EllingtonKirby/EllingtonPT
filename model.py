INF = 1e10
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
from math import sqrt

class SelfAttention(nn.Module):
  def __init__(self, embed_dim, scaled_attention=False) -> None:
    super().__init__()
    self.embed_dim = embed_dim
    self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.activation = nn.Softmax(dim=-1)
    self.scaled_attention = scaled_attention
  
  def forward(self, input):
    Q = self.w_q(input)
    K = self.w_k(input)
    V = self.w_v(input)
    self_attention = torch.bmm(Q, K.mT)
    if self.scaled_attention:
      attention_scaling = sqrt(self.embed_dim)
      softmax = self.activation(self_attention / attention_scaling)
    else:
      softmax = self.activation(self_attention)
    output = torch.bmm(softmax, V)
    return output, self_attention

class FeedForward(nn.Module):
  def __init__(self, embed_dim,) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(in_features=embed_dim, out_features=3072)
    self.non_linear = nn.LeakyReLU()
    self.linear_2 = nn.Linear(in_features=3072, out_features=embed_dim)
  
  def forward(self, input):
    input = self.linear_1(input)
    input = self.non_linear(input)
    return self.linear_2(input)

class BasicTransformerLayer(nn.Module):
  def __init__(self, embed_dim) -> None:
    super().__init__()
    self.attention = SelfAttention(embed_dim)
    self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
    self.feed_forward = FeedForward(embed_dim)
    self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)

  def forward(self, input):
    weights, _ = self.attention(input)
    normalized = self.layer_norm_1(weights + input)
    fed = self.feed_forward(normalized)
    return self.layer_norm_2(fed + normalized)

INF = 1e10

class MaskedTransformerLayer(nn.Module):
  def __init__(self, embed_dim) -> None:
    super().__init__()
    self.attention = MaskedSelfAttention(embed_dim)
    self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
    self.feed_forward = FeedForward(embed_dim)
    self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)

  def forward(self, input, attention_mask):
    weights, _ = self.attention(input, attention_mask)
    normalized = self.layer_norm_1(weights + input)
    fed = self.feed_forward(normalized)
    return self.layer_norm_2(fed+normalized)
  
class MaskedSelfAttention(nn.Module):
  def __init__(self, embed_dim, scaled_attention=False) -> None:
    super().__init__()
    self.embed_dim = embed_dim
    self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.activation = nn.Softmax(dim=-1)
    self.scaled_attention = scaled_attention
  
  def forward(self, input, attention_mask):
    Q = self.w_q(input)
    K = self.w_k(input)
    V = self.w_v(input)
    self_attention = torch.bmm(Q, K.mT)
    padding_mask = (1.0 - attention_mask) * -INF
    self_attention += padding_mask[:, None, :]
    if self.scaled_attention:
      attention_scaling = sqrt(self.embed_dim)
      softmax = self.activation(self_attention / attention_scaling)
    else:
      softmax = self.activation(self_attention)
    output = torch.bmm(softmax, V)
    return output, self_attention

class CausalTransformerLayer(nn.Module):
  def __init__(self, embed_dim) -> None:
    super().__init__()
    self.attention = CausalSelfAttention(embed_dim)
    self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
    self.feed_forward = FeedForward(embed_dim)
    self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)

  def forward(self, input, attention_mask):
    weights, _ = self.attention(input, attention_mask)
    normalized = self.layer_norm_1(weights + input)
    fed = self.feed_forward(normalized)
    return self.layer_norm_2(fed+normalized)
  
class CausalSelfAttention(nn.Module):
  def __init__(self, embed_dim, scaled_attention=False) -> None:
    super().__init__()
    self.embed_dim = embed_dim
    self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.activation = nn.Softmax(dim=-1)
    self.scaled_attention = scaled_attention
  
  def forward(self, input, attention_mask):
    Q = self.w_q(input)
    K = self.w_k(input)
    V = self.w_v(input)
    self_attention = torch.bmm(Q, K.mT)
    padding_mask = (1.0 - attention_mask)
    causal_mask = torch.triu(torch.ones_like(self_attention), diagonal=1)
    mask = causal_mask + padding_mask[:, None, :]
    self_attention += mask * -INF
    if self.scaled_attention:
      attention_scaling = sqrt(self.embed_dim)
      softmax = self.activation(self_attention / attention_scaling)
    else:
      softmax = self.activation(self_attention)
    output = torch.bmm(softmax, V)
    return output, self_attention
