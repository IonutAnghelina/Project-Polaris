import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
class Transformer(nn.Module):

    def __init__(self, vocab_size : int, target_vocab_size : int, source_seq_len : int, target_seq_len : int, embedding_size : int = 512, no_encoder_blocks : int = 6, no_decoder_blocks : int = 6, expansion_factor : int = 4,
                no_heads : int = 8, p_dropout : float = 0.1):

        super(Transformer, self).__init__()
        """
        The main Transformer class. Its main focus is sequence-to-sequence modelling.

        The constructor receives as input

        vocab_size - int, the size of the input vocabulary

        target_vocab_size - int, the size of the output vocabulary

        source_seq_len - int, the size of the input sequence

        target_seq_len - int, the size of the output sequence 

        embedding_size - int, the size of the embedding space 
        
        no_encoder_blocks - int, the number of encoder blocks

        no_decoder_blocks - int, the number of decoder blocks 

        expansion_factor - int, The ratio between the size of the hidden layer and the size of the input in the FeedForward submodule

        no_heads - int, the number of heads in the Multi Headed Attention

        p_dropout - float, the probability of dropout in the FeedForward network
        """

        self.vocab_size = vocab_size
        self.source_seq_len = source_seq_len 
        self.target_seq_len = target_seq_len
        self.target_vocab_size = target_vocab_size
        self.embedding_size = embedding_size
        self.no_encoder_blocks = no_encoder_blocks
        self.no_decoder_blocks = no_decoder_blocks
        self.expansion_factor = expansion_factor
        self.no_heads = no_heads
        self.p_dropout = p_dropout

        self.encoder = Encoder(self.source_seq_len, self.vocab_size, self.embedding_size, self.no_encoder_blocks, self.expansion_factor, self.no_heads, self.p_dropout)
        self.decoder = Decoder(self.target_seq_len, self.target_vocab_size, self.embedding_size, self.no_decoder_blocks, self.expansion_factor, self.no_heads, self.p_dropout)

    def make_mask(self, target : torch.Tensor) -> torch.Tensor:

        """
        Builds the masked self attention mask

        Receives as input

        target - [batch_size, target_seq_len]

        Returns

        mask - [batch_size, target_seq_len, target_seq_len]

        """
     
        batch_size = target.shape[0]
     
        target_len = target.shape[1]
     
        mask = torch.tril(torch.ones(target_len, target_len)).expand(batch_size, self.no_heads, target_len, target_len)
       
        return mask

    def make_cross_attention_mask(self, source : torch.Tensor, target : torch.Tensor) -> torch.Tensor:

        """
        Builds a padding mask for cross-attention

        Receives as input 

        source - [batch_size, source_seq_len]
        target - [batch_size, target_seq_len]

        Returns

        mask - [batch_size, target_seq_len, source_seq_len], the padding mask
        """
        batch_size = source.shape[0]
        source_len = source.shape[1]
        target_len = target.shape[1]

        source_matrix = (source!=0)
        source_matrix = source_matrix.unsqueeze(1).unsqueeze(1).expand(batch_size, self.no_heads, target_len, source_len)

        padding_mask = source_matrix.int()
    

        return padding_mask
        
    def forward(self, source : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Applies the encoder and then decoder blocks successively

        Receives as input:

        source - [batch_size, source_seq_len]
        target - [batch_size, target_seq_len]

        Returns:

        output - [batch_size, target_seq_len, target_vocab_size] 
        """
        encoder_output = self.encoder(source)
       
        target_mask = self.make_mask(target)
        
        cross_mask = self.make_cross_attention_mask(source, target)
        
        return self.decoder(target, encoder_output, target_mask, cross_mask)

    def decode(self, target_seq_len : int, source : torch.Tensor, show_steps = False, target : torch.Tensor = torch.Tensor([[1]])) -> torch.Tensor:
        
        """
        Uses the pretrained transformer for a particular instance

        Receives as input

        target_seq_len - int, the desired length of the output 
        source - [batch_size, source_seq_len], the token sequence in the input
        target - [batch_size, 1], the initial target output 

        Returns:

        out - a list of token indices corresponding to the correct translation of the input 
        """

        with torch.no_grad():

            current_length = target.shape[1]
            
            batch_size = source.shape[0]

            output_labels = torch.ones(batch_size, 1) #The <SOS> token
            
            output_labels = F.pad(output_labels, (0, target_seq_len - current_length), value=0).int()
            
            encoder_output = self.encoder(source)

            out = output_labels
            
            currentPredictingPosition = current_length 

            for i in range(current_length, target_seq_len):
                
                decoder_mask = self.make_mask(out)
                cross_mask = self.make_cross_attention_mask(source, out)
             
                res = self.decoder(out, encoder_output, decoder_mask, cross_mask)
              
                res = res.argmax(dim = -1) #No need for softmax
                
                newTokens = res[:, currentPredictingPosition - 1]

                out[:, currentPredictingPosition] = newTokens

                currentPredictingPosition += 1
              
                if show_steps:
                    print(f"Current out is: {out}")
              
               
                

            return out