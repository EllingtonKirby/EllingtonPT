from model import BasicTransformerLayer, MaskedTransformerLayer, CausalTransformerLayer
import torch

def test_basic_transformer():
  d_embed = 8
  batch_size = 4
  seq_length = 6
  for seq_length in [1, 2, 4, 6, 10]:
      test_input = torch.randn(batch_size, seq_length, d_embed)
      transformer = BasicTransformerLayer(d_embed)
      out = transformer(test_input)
      assert out.shape == test_input.shape

def make_random_attention_mask(batch_size, seq_length):
    attention_mask = torch.ones(batch_size, seq_length)
    max_ind = torch.randint(0, seq_length, (batch_size, 1))
    indices = torch.arange(seq_length)
    attention_mask[indices >= max_ind] = 0
    return attention_mask

def test_masked_attention():
  d_embed = 8
  batch_size = 4
  seq_length = 6
  for seq_length in [2, 4, 6, 10]:
      test_input = torch.randn(batch_size, seq_length, d_embed)
      transformer = MaskedTransformerLayer(d_embed)
      attention_mask = make_random_attention_mask(batch_size, seq_length)
      out = transformer(test_input, attention_mask)
      assert out.shape == test_input.shape

def test_causal_attention():
  d_embed = 8
  batch_size = 1
  seq_length = 6
  for seq_length in [2, 4, 6, 10]:
      attention_mask = make_random_attention_mask(batch_size, seq_length)
      test_input = torch.randn(batch_size, seq_length, d_embed)
      transformer = CausalTransformerLayer(d_embed)
      out = transformer(test_input, attention_mask)
      assert out.shape == test_input.shape


if __name__=='__main__':
   print('-'*100)
   print('Testing Basic Transformer')
   test_basic_transformer()
   print('Test Passed')
   print('-'*100)
   print('Testing Masked Transformer')
   test_masked_attention()
   print('Test Passed')
   print('-'*100)
   print('Testing Causal Transformer')
   test_causal_attention()
   print('Test Passed')
   print('-'*100)
