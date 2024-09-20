from GPT.decoder_block import DecoderBlock
from Transformer.embedding_layer import EmbeddingLayer
from Transformer.positional_encoder import PositionalEncoder
import torch
import torch.nn as nn

class GPTDecoder(nn.Module):

    def __init__(self, seq_len : int = 768, target_vocab_size : int =  40000, embedding_dim : int = 768, no_heads : int = 12, expansion_factor : int = 4, dropout_rate : float = 0.1, no_decoder_blocks : int = 12):

        """
        This layer passes the (key, query, value) tensors through all the decoder 
        Blocks in the transformer 

        The constructor receives as input

        seq_len - the length of the input sequence
        target_vocab_size - the size of the vocabulary
        embedding_dim - the embedding dimension
        no_heads - the number of heads in the self attention layer
        expansion_factor - the ratio between the sizes of the fully connected layers in the Feed-Forward Network
        dropout_rate - the dropout probability in the Feed-Forward Network
        no_blocks - the number of decoder blocks in the stack
        """
        super(GPTDecoder, self).__init__()

        self.seq_len = seq_len
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.no_heads = no_heads 
        self.expansion_factor = expansion_factor
        self.dropout_rate = dropout_rate
        self.no_decoder_blocks = no_decoder_blocks
        self.decoder_stack = nn.ModuleList([DecoderBlock(self.embedding_dim, self.no_heads, self.expansion_factor, self.dropout_rate) for i in range(self.no_decoder_blocks)])
        self.embedding_layer = EmbeddingLayer(self.target_vocab_size, self.embedding_dim)
        self.positional_embedder = PositionalEncoder(self.seq_len, self.embedding_dim)

    def make_mask(self, target : torch.Tensor) -> torch.Tensor:
        """
        This function creates the attention mask for the decoder

        Receives as input:

        target - [batch_size, seq_len] - the input tensor

        Returns:

        mask - [batch_size, no_heads, seq_len, seq_len]

        """
        
        batch_size = target.shape[0]
     
        target_len = target.shape[1]
     
        mask = torch.tril(torch.ones(target_len, target_len)).expand(batch_size, self.no_heads, target_len, target_len)
       
        return mask
        
    def forward(self, x : torch.Tensor):
        
        """
        This function passes the input tensor through each decoder in the stack

        Receives as input:

        x - [batch_size, seq_len]

        Returns:

        res - [batch_size, seq_len, embedding_size]
        """
        mask = self.make_mask(x)

        inputTensor = self.embedding_layer(x)
        
        res = self.positional_embedder(inputTensor)
        
        for decoderBlock in self.decoder_stack:
            res = decoderBlock(res, res, res, mask)

        return res


