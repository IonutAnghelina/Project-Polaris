from transformer import Transformer
import torch 

model = Transformer(vocab_size = 100, target_vocab_size = 100, source_seq_len = 10, target_seq_len = 10, embedding_size = 512)
source = torch.Tensor([[1,1,2,3,4,5,0,0,0,0]]).type(torch.LongTensor)

print(model.decode(10,source))