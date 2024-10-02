import torch
from transformer import Transformer
from utils import setup_logger
import logging


# Initialise useful variables
tot_ep = 100 # n. of training step
vocab_size = 50 # total n. of token
batch_s = 50 # batch size
input_seq_s = 10 # length of the input sequence
target_seq_s = 12 # length of the ouput sequence
max_seq_len = 20 # max sequence length (for positional encoding)
embedding_s= 512 # size of the embedding space
n_heads= 4 # n. of attention heads
ln_rate = 5e-4 # learning rate

# Select correct device
if torch.cuda.is_available():
    dev='cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
    dev='mps'
else:
    dev='cpu'


assert target_seq_s <= max_seq_len and input_seq_s <= max_seq_len, "Sequence size larger than max sequence size"

# set-up logger
setup_logger()

# Generate random input and target sequences by initialising a list of indexes for each batch 
#e.g., each index could denote a different word, where the input sequence is words in english while the target sequence is the translation in italian.
x = torch.randint(1,vocab_size-2,(batch_s,input_seq_s)).to(dev)
target_x = torch.randint(1,vocab_size-2,(batch_s,target_seq_s)).to(dev)

# --- To test inference performance and ability to stop, I use vocab_size-1 and vocab_size-2 as the  start and end token respectively
start_token = vocab_size-2
end_token = vocab_size -1

# Add start token at beginning of sequence 
x[:,0] = start_token
target_x[:,0] = start_token
# Add end token at end of sequence 
x[:,-1] = end_token
target_x[:,-1] = end_token

# Initialise overall stransformer architecture
transformer = Transformer(input_vocab_s=vocab_size, target_vocab_s=vocab_size, max_seq_len=max_seq_len, embedding_s=embedding_s, n_heads=n_heads, ln_rate=ln_rate).to(dev)

# training:
for ep in range(tot_ep):
    # exclude last element of target seq since don't wanna predict next token for last element
    transformer_output = transformer(input_seq=x, target_seq=target_x[:,:-1])
    # shift target by one since want to predict next token
    loss = transformer.update(transformer_output.contiguous().view(-1, vocab_size), target_x[:,1:].contiguous().view(-1))
    logging.info(f" *** Episode: {ep} | Loss: {loss} *** ")

# Try inference by selecting sequence from the first batch 
# this is just to try the inference step since all inputs were random, 
# so we don't expect any meaningful learning beyond some memorisation
x = x[0:1,:]
target_x = target_x[0:1,:]
predictions = transformer.inference(input_seq=x, start_token=start_token, end_token=end_token, max_decoder_steps=max_seq_len)

logging.info(f" *** Predicted sequence: {predictions}")
logging.info(f" *** Target sequence: {target_x}")
