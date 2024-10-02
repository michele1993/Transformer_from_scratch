import torch
from MultiHeadAttention import MultiHeadAttention, FF_network
from embedding import PositionalEmbedding, Embedding
import torch.nn as nn

class DecoderBlock(nn.Module):
    """ Implement all the decoder operations inside one encoder module. """
        
    def __init__(self, embedding_s, n_heads):

        super().__init__()

        # Initialise 1st multi-head self-attention (SA) layer
        self.masked_multiHead_attention = MultiHeadAttention(embedding_s=embedding_s, n_heads=n_heads)

        # Initialise 1st normalisation layer
        self.norm_l1 = nn.LayerNorm(embedding_s) 

        # Initialise 1st multi-head self-attention (SA) layer
        self.cross_multiHead_attention = MultiHeadAttention(embedding_s=embedding_s, n_heads=n_heads)

        # Initialise 2nd normalisation layer
        self.norm_l2 = nn.LayerNorm(embedding_s) 

        # Initialise FeedForward layer to merge SA output
        self.l1 = FF_network(embedding_s)

        # Initialise final normalisation layer
        self.norm_l3 = nn.LayerNorm(embedding_s)

    def forward(self, x, encoder_output, mask):
        """ 
            Args:
                x: the output embedding
                encoder_output: the ouput of the encoder
                mask: to hide following elements in the output seq
        """
        # Masked multi-head attention
        attention = self.masked_multiHead_attention(Q=x,K=x,V=x, mask=mask)

        # Norm layer with residual connection
        x = self.norm_l1(attention + x)

        # Compute the cross-attention output between the decoder and encoder outputs
        # using the decoder to query the encoder output for the useful information
        cross_attention = self.cross_multiHead_attention(Q=x, K=encoder_output, V=encoder_output)

        # Norm layer with residual connection
        x = self.norm_l2(cross_attention + x)

        # Norm layer with residual connection
        ff_output = self.l1(x)

        # final norm layer with residual connection
        return self.norm_l3(ff_output + x)
