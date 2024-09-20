import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim : int = 512, no_heads : int = 8):


        """
        This layer projects the key, query and value
        It then computes the attention mask and the output matrix
        It works in a multi-head fashion. Each head will process a slice of the embedding vectors

        The constructor receives as input:

        embedding_dim - int, the embedding dimension of the keys, queries and values
        no_heads - int, the number of heads in which the vectors will be split

        """
        super(MultiHeadAttention,self).__init__()
        self.embedding_dim = embedding_dim
        self.no_heads = no_heads
    
        self.split_size = self.embedding_dim // self.no_heads #The input size of the tensor that will be received through each head
  
        self.WQuery = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False) #The projection matrices for the three spaces
        self.WKey = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False)
        self.WValue = nn.Linear(self.embedding_dim, self.embedding_dim, bias = False)
        self.WOut = nn.Linear(self.no_heads * self.split_size, self.embedding_dim, bias = True) #The actual values are needed, not just similarities, so we include bias

    def forward(self, key : torch.Tensor, query : torch.Tensor, value : torch.Tensor, mask : torch.Tensor = None, padding_mask : torch.Tensor= None) -> torch.Tensor: 
        
        """
        Receives

        key, value - [batch_size, seq_length, embedding_dim]
        
        query - [batch_size, query_seq_length, embedding_dim]

        The tensors that will be projected into the three spaces

        mask - [batch_size, key_seq_length, key_seq_length]

        The boolean mask that will be used in the case of masked self-attention

        padding_mask - [batch_size, query_seq_length, seq_length]

        The padding mask which will be used in order for padding tokens to not attend to other actual tokens

        Returns

        val - [batch_size, query_seq_len, embedding_dim] 

        The new embeddings refined through attention

        """

        batch_size = key.size(0)
        seq_length = key.size(1)

        k = self.WKey(key)
        q = self.WQuery(query)
        v = self.WValue(value)
        
        k = k.view(batch_size, seq_length, self.no_heads, self.split_size) #We split the information into multiple vectors, one for each head

        seq_length_for_query = q.shape[1] #It can be different to seq_length when dealing with cross-attention

        q = q.view(batch_size, seq_length_for_query, self.no_heads, int(self.split_size))

        v = v.view(batch_size, seq_length, self.no_heads, int(self.split_size))

        k = k.transpose(1,2) # [batch_size, no_heads, seq_length, split_size]. We move the batch dimensions first
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # https://pytorch.org/docs/stable/generated/torch.matmul.html - For broadcasting explainations

        k_transposed = k.transpose(2 , 3) #Now k is [batch_size, no_heads, split_size, seq_length]

        attention_mask = torch.matmul(q, k_transposed)

        if mask is not None:
            attention_mask[mask == 0] = -float('inf')
    
        #if padding_mask is not None:
        #    attention_mask[padding_mask == 0] = -float('inf')
        
        attention_mask = attention_mask / ((self.split_size)**0.5) # The result will be [batch_size, no_heads, seq_len_query, seq_len]

        attention_mask = F.softmax(attention_mask, dim = -1) #Normalizing on the columns
        
        val = torch.matmul(attention_mask, v) # [batch_size, no_heads, seq_len_query, split_size]
     
        val = val.transpose(1,2).contiguous() # [batch_size, seq_len_query, num_heads, split_size]

        val = val.view(batch_size, seq_length, self.embedding_dim) #[batch_size, seq_len_query, embedding_dim]

        output = self.WOut(val) #[batch_size, seq_len_query, embedding_dim]

        
        return output