import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
from encoder_block import EncoderBlock

class DecoderBlock(nn.Module):

    def __init__(self, embedding_dim : int = 512, no_heads : int = 8 , expansion_factor : int = 4, p_dropout : int = 0.1):

        """
        This layer passes the (query, key, value) set through an encoder block

        The constructor receives as input

        embedding_dim - int, the dimension of the embeddings in query, key, value
        no_heads - int, the number of heads in Multi Head Attention
        expansion_factor - int, The ratio between the size of the hidden layer and the size of the input in the FeedForward submodule
        p_dropout - float, the dropout probability in the fully connected layers
        """
        
        super(DecoderBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.no_heads = no_heads
        self.expansion_factor = expansion_factor
        self.p_dropout = p_dropout
        self.selfAttentionLayer =  MultiHeadAttention(self.embedding_dim, self.no_heads)
        self.firstLayerNorm = nn.LayerNorm(self.embedding_dim)
        self.dropout1 = nn.Dropout(self.p_dropout)
        self.encoderBlock = EncoderBlock(self.embedding_dim, self.expansion_factor, self.no_heads, self.p_dropout)
    
    def forward(self, decoder_key : torch.Tensor, decoder_query : torch.Tensor, decoder_value : torch.Tensor, encoder_key : torch.Tensor, encoder_query : torch.Tensor, encoder_value : torch.Tensor, mask : torch.Tensor = None, padding_mask : torch.Tensor = None, cross_attention_mask : torch.Tensor = None) -> torch.Tensor:

        """
        Receives as input:

        decoder_key, decoder_value - [batch_size, target_seq_length, embedding_dim]
        
        decoder_query - [batch_size, target_query_seq_length, embedding_dim]

        encoder_key, encoder_value - [batch_size, source_seq_length, embedding_dim]
        
        encoder_query - [batch_size, source_query_seq_length, embedding_dim]


        The tensors that will be projected into the three spaces

        mask - [batch_size, key_seq_length, key_seq_length]

        The boolean mask that will be used in the case of masked self-attention

        padding_mask - [batch_size, query_seq_length, seq_length]

        The padding mask which will be used in order for padding tokens to not attend to other actual tokens

        cross_attention_mask - [batch_size, query_seq_length, seq_length]

        The mask used to mask padding tokens in cross attention

        Returns

        val - [batch_size, query_seq_len, embedding_dim] 

        The result obtained by passing the self-attention results through the Fully Connected and LayerNorm layers

        """

        selfAttentionOutput = self.selfAttentionLayer(decoder_key, decoder_query, decoder_value, mask, None)
       
        attentionDropoutResNorm = self.firstLayerNorm(decoder_value + self.dropout1(selfAttentionOutput))

        crossAttentionOutput = self.encoderBlock(encoder_key, attentionDropoutResNorm, encoder_value, None, cross_attention_mask)
    
        return crossAttentionOutput