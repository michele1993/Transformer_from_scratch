import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    """ Embedding class to convert each element in the input sequence (e.g., words) into a corresponding embedding vector (providing a 'relational' representation across elements)"""

    def __init__(self, vocab_size, embedding_s):
        """
            Args:
                vocab_size: n. of total unique words/elements in entire data-set (i.e., need a different index for each unique element)
                embedding_s: size of the vector embedding
        """

        super().__init__()

        # NOTE: In PyTorch, the nn.Embedding class is a simple look-up table from indides (e.g., of words) to vector embeddings. 
        # This look-up table can be inialised with known embeddings (e.g., work2vec weights) or it can be learned with backprop.
        # (see https://discuss.pytorch.org/t/how-does-nn-embedding-work/88518/2 for a nice explanation)
        # Specifically, it creates a [vocab_size x embedding_s] matrix, where each (input) index is used to index 
        # a row of the matrix, providng the corresponding  embedding representation.
        self.embedding = nn.Embedding(vocab_size, embedding_s)

    def forward(self, x):
        """
        Args:
            x: raw input sequence (e.g., word)
        Returns:
            out: embedding vector
        """
        return self.embedding(x)

class PositionalEmbedding(nn.Module):
    """ Create a positional embedding to be added to the embedding vector to encode each element position in the sequence """

    def __init__(self, embedding_s, max_seq_len): 
        """
            Args:
                embedding_s: size of the vector embedding
                max_seq_len: maximum length for an input sequence
        """
        super().__init__()


        # Positional encoding based on "Attention is all you need" paper
        # using sine and cosine functions
        pe = torch.zeros(max_seq_len, embedding_s)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_s, 2).float() * -(math.log(10000.0) / embedding_s)).unsqueeze(0)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # NOTE: register_buffer allows you to store params in state_dict without being return in model.parameters() 
        # so that these parameters won't be updated by the optimiser (this is the key difference to register_parameter) 
        # beyond saving purposes, having params in the state_dict is useful as all these params will be pushed to the device, if called on the parent model
        # model.to(dev)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """ 
        Add  the positional encoding to the embedding vector
        Args:
            x: embedding vector [batch_s, seq, embedding_s]
        Return:
            the embedding vector with the positional encoding
        """
        return x + self.pe[:, :x.size(1)]
