import torch
from MultiHeadAttention import MultiHeadAttention, FF_network
from embedding import PositionalEmbedding, Embedding
import torch.nn as nn

class EncoderBlock(nn.Module):
    """ 
        Implement all the encoder operations inside one encoder module. 
        NOTE: the embedding operations are NOT perform inside the encoder block since
        we may want to use multiple parallel encoder blocks, which should already receives
        the embedded input
    """

    def __init__(self, embedding_s, n_heads):
        """ 
        Args:
                embedding_s: input embedding size,
                vocab_size: n. of total unique words/elements in entire data-set (i.e., need a different index for each unique element)
                max_seq_len: maximum length for an input sequence
                n_heads: n. of attention heads
        """

        super().__init__()

        # Initialise embedding layer
        #self.embedding = Embedding(vocab_size=vocab_size, embedding_s=embedding_s)
        # Initialise positional embedding layer
        #self.pos_embedding = PositionalEmbedding(embedding_s=embedding_s, max_seq_len=max_seq_len)

        # Initialise multi-head self-attention (SA) layer
        self.multiHead_attention = MultiHeadAttention(embedding_s=embedding_s, n_heads=n_heads)

        # Initialise 1st normalisation layer
        self.norm_l1 = nn.LayerNorm(embedding_s) 

        # Initialise FeedForward layer to merge SA output
        self.l1 = FF_network(embedding_s)

        # Initialise final normalisation layer
        self.norm_l2 = nn.LayerNorm(embedding_s)

    def forward(self, x, mask=None):
        """ 
        Args:
            x: embedding vector [batch_s, seq, embedding_s]
            mask: input mask to mask padding tokens input sequence (don't want to pay attention to padding tokens)
        Return:
            The encoded input sequence
        """

        # create the embedding for the input batch
        #embed_vector = self.embedding(x)

        # Add positional encoding to embedding
        #embed_vector = self.pos_embedding(embed_vector)

        # Apply multi-head attention
        attention = self.multiHead_attention(Q=x, K=x, V=x, mask=mask)

        # Apply norm layer with skip connection
        x = self.norm_l1(attention + x)

        # Apply feedforward layer
        ff_output = self.l1(x)

        # Apply second norm layer with skip connection
        return self.norm_l2(ff_output + x)


