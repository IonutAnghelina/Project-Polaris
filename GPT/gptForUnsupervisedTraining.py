from GPT.decoder_block import DecoderBlock
from Transformer.embedding_layer import EmbeddingLayer
from Transformer.positional_encoder import PositionalEncoder
from GPT.gptDecoder import GPTDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
class GPTForUnsupervisedTraining(nn.Module):

    def __init__(self, baseModel : GPTDecoder, vocab_size : int = 40000):

        """
        This class adds a final layer which converts the output of the GPT model from

        [batch_size, seq_len, embedding_dim] to [batch_size, seq_len, vocab_size]

        So that the model can be trained using Next Token Prediction Objective

        The constructor receives as input

        baseModel - GPTDecoder, the stack of decoders which represent the base of the GPT model

        embedding_dim - int, the dimension of the embedding space 

        vocab_size - int, the dimension of the vocabulary 

        """
        super(GPTForUnsupervisedTraining, self).__init__()
        self.baseModel = baseModel 
        self.embedding_dim = baseModel.embedding_dim
        self.vocab_size = vocab_size

        self.fc = nn.Linear(self.embedding_dim, self.vocab_size)

    def decode(self, target_seq_len : int, inputTensor: torch.Tensor = torch.Tensor([[0]]), show_steps : bool = True, temperature : float = 1.0, greedy_decoding = True) -> torch.Tensor:
        
        """
        
        This function generates an input based on a prompt
        Receives as input:

        target_seq_len - the maximum length of the desired result

        inputTensor - [1, current_length], the prompt

        """
        with torch.no_grad():

            current_length = inputTensor.shape[1]
            
            inputTensor = F.pad(inputTensor, (0, target_seq_len - current_length), value = 0).int()

            currentPredictingPosition = current_length

            for i in range(current_length, target_seq_len):
                decoder_mask = self.baseModel.make_mask(inputTensor)

                res = self.fc(self.baseModel(inputTensor))

                tok = None
                if not greedy_decoding:
                    res = nn.Softmax(dim = -1)(res / temperature)
                    tok = torch.multinomial(res[0, currentPredictingPosition - 1], 1).item()
                else:
                    res = nn.Softmax(dim = -1)(res / temperature)
                    res = res.argmax(dim = -1)
                    tok = res[0, currentPredictingPosition-1]
                    
                newTokens = tok

                inputTensor[0, currentPredictingPosition] = newTokens

                currentPredictingPosition += 1
              
                if show_steps:
                    print(f"Current out is: {inputTensor}")
  
            return inputTensor
        
    def forward(self, inputTensor : torch.Tensor) -> torch.Tensor:
        
        """
        The function receives as input:

        inputTensor - [batch_size, seq_len]

        The function returns:

        res - [batch_size, seq_len, vocab_size]
        
        """
        res = self.baseModel(inputTensor)

        return self.fc(res)
    

