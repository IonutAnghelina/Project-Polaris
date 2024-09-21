from Transformer.multi_head_attention import MultiHeadAttention
import torch.nn as nn


class DecoderBlock(nn.Module):

    def __init__(self, embedding_dim : int = 768, no_heads : int = 12, expansion_factor : int = 4, dropout_rate : float = 0.1):
        

        """
        This layer passes the (query, key, value) set through a decoder block

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
        self.dropout_rate = dropout_rate

        self.selfAttentionLayer = MultiHeadAttention(self.embedding_dim, self.no_heads)
        self.firstLayerNorm = nn.LayerNorm(self.embedding_dim)
        self.secondLayerNorm = nn.LayerNorm(self.embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * self.expansion_factor),
            nn.GELU(),
            nn.Linear(self.embedding_dim * self.expansion_factor, self.embedding_dim),
            nn.Dropout(self.dropout_rate)   
        )
        
        self.dropoutLayer = nn.Dropout(self.dropout_rate)


    def forward(self, key : torch.Tensor, query : torch.Tensor, value : torch.Tensor, mask : torch.Tensor = None) -> torch.Tensor:

        """
        Receives as input:

        key - [batch_size, seq_len, embedding_dim] - the values that will be projected into the key space
        query - [batch_size, seq_len, embedding_dim] - the values that will be projected into the query space
        value - [batch_size, seq_len, embedding_dim] - the values that will be proected into the value space

        mask - [batch_size, no_heads, seq_len, seq_len]

        Returns:

        res - [batch_size, seq_len, embedding_dim]

        """
        attentionOutput = self.selfAttentionLayer(key = key, query = query, value = value, mask = mask)

        x = (value + attentionOutput)
        x = self.firstLayerNorm(x)

        fc1 = self.fc(x)
        res = (x + fc1)
        res = self.secondLayerNorm(res)

        return res



