import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_s, n_heads):
        """ 
        Args:
                embedding_s: input embedding size,
                n_heads: n. of attention heads
        """

        super().__init__()

        assert embedding_s % n_heads == 0, "Input size must be divisible by n. of attention heads"

        self.n_heads = n_heads
        self.chunk_s = embedding_s // self.n_heads

        self.Q_weight = nn.Linear(embedding_s, embedding_s, bias=False)
        self.K_weight = nn.Linear(embedding_s, embedding_s, bias=False)
        self.V_weight = nn.Linear(embedding_s, embedding_s, bias=False)

        # FInal linear layer to unify output of each head
        self.unifyHeads = nn.Linear(embedding_s,embedding_s)

    def _chunk_heads(self, x):
        """ Chunk Q,K,V appropriately to apply different attention heads """

        # Extract useful quantities 
        batch_s = x.shape[0]
        seq_s = x.shape[1]
            
        # --- First: split the input dimension across seprate heads
        x = x.view(batch_s, seq_s, self.n_heads, self.chunk_s)

        # --- Second: to apply batch matrix multiplication can treat the different
        # heads as extra batch elements since they are independent

        # to do so first need to transpose n_heads with seq_s so can rearange
        # n_heads into batch dim
        x = x.transpose(1,2).contiguous()

        # return inputs with n_head inserted into batch elements
        return x.view(batch_s*self.n_heads, seq_s, self.chunk_s)


    def forward(self, Q, K, V, mask=None):
        """
        Note, pass Q,K,V as separate inputs so that in decoder block can use K and V from encoder and Q from decoder
        Args:
            Q: query (for the encoder block this is the input embedding)
            K: key (for the encoder block this is the input embedding)
            V: value (for the encoder block this is the input embedding)
            mask: mask used to mask the following elements in the sequence for the decoder
        """

        # Extract some useful variables
        batch_s = K.size(0)
        seq_s = K.size(1)
        # query dimension can change in decoder during inference.
        # so we cant take general seq_length, (otherwise just same as seq_lenght)
        query_seq_s = Q.size(1)

        # Compute overall query, key and value
        Q = self.Q_weight(Q)
        K = self.K_weight(K)
        V = self.V_weight(V)

        # Now chunk Q,K,V across attention head and dumpt them into batch size
        # so that can apply matrix batch multiplication
        Q = self._chunk_heads(Q)
        K = self._chunk_heads(K)
        V = self._chunk_heads(V)

        # Compute one attention weight for relation between each seq element
        attention_score = Q @ K.transpose(1,2) # [batch * n_heads, query_seq, chunk_s] @ [batch_s * n_heads, chunk_s, seq] = [..., query_seq, seq] 

        # Apply mask if necessary
        if mask is not None:
            # set attention score to a super small value for masked elements
            # in practice telling the transformer not to 'pay attention' to them
            attention_score = attention_score.masked_fill(mask == 0, float("-1e20"))

        # Normalised dot product:
        norm_attention_score = attention_score / self.chunk_s**(1/2)
        # Softmax score
        attention_weight = torch.softmax(attention_score, dim=-1)
        # Compute values
        values = attention_weight @ V # [batch * n_heads, query_seq, seq] @ [batch * n_heads, seq, chunk_s] = [..., query_seq, chunk_s]

        # Recombine the output of each head into the embedding dimension
        # NOTE: here the sequence length is based on Q (i.e., see above computations)
        values = values.view(batch_s, self.n_heads, query_seq_s, self.chunk_s).transpose(1,2).contiguous() # 1st, need to transpose back n_heads and seq
        values = values.view(batch_s, query_seq_s, self.n_heads * self.chunk_s) # 2nd, unify head outputs into final dim
            
        # finally aplly linear operation
        return self.unifyHeads(values)


class FF_network(nn.Module):

    def __init__(self, embedding_s, h_units=116):
        """ 
        Args:
                embedding_s: input embedding size
        """
        
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(embedding_s, h_units),
            nn.ReLU(),
            nn.Linear(h_units,embedding_s)
        )

    def forward(self, x):
        return self.ff(x)
