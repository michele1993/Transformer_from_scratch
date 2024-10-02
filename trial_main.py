import torch
from MultiHeadAttention import MultiHeadAttention
from embedding import Embedding, PositionalEmbedding
from encoder import EncoderBlock
from decoder import DecoderBlock

vocab_size = 100
batch_s = 34
seq_s = 20
embedding_s= 512
n_heads=4

#trial = torch.tril(torch.ones(batch_s, seq_s, seq_s))
#print(trial)

# Initialise word embedding and positional embedding
embedding = Embedding(vocab_size=vocab_size, embedding_s=embedding_s)
pos_embedding = PositionalEmbedding(embedding_s=embedding_s, max_seq_len=20)

# Initialise a list of indexes for a single input batch, for instance each index could denote a different word
x = torch.randint(1,vocab_size,(batch_s,seq_s))
target_x = torch.randint(1,vocab_size,(batch_s,seq_s))


# create the embedding for the input batch
embed_inpt = embedding(x)

# Add positional encoding to embedding
embed_inpt = pos_embedding(embed_inpt)

##  ------ Test attention mechanism -------
attention = MultiHeadAttention(embedding_s=embedding_s, n_heads=n_heads)
y =  attention(Q=embed_inpt, K=embed_inpt, V=embed_inpt)
#print(y.shape)
## -----------------------

##  ------ Test encoder block -------
encoder = EncoderBlock(embedding_s=embedding_s, n_heads=n_heads)
encoder_output = encoder(embed_inpt)
#print(encoder_output.shape)
## -----------------------

## -- Test decoder block -----
target_emb = embedding(target_x)
target_emb = pos_embedding(target_emb)
target_seq_s = target_emb.shape[-2]

decoder = DecoderBlock(embedding_s=embedding_s, n_heads=n_heads) 
decoder_output_s = []

for i in range(target_seq_s): 
    mask = torch.ones(n_heads*batch_s,target_seq_s, target_seq_s)
    mask[:,i+1:] = 0
    decoder_output = decoder(x=target_emb, encoder_output=encoder_output, mask=mask)
    print(decoder_output.shape)
    exit()
    decoder_output_s.append(decoder_output)
