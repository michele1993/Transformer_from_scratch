import torch
from transformer import Transformer
from utils import setup_logger
from utils import generate_randomData_batch
import logging


# Initialise useful variables
tot_ep = 50 # n. of training step
vocab_size = 50 # total n. of token
max_seq_len = 20 # max sequence length (for positional encoding)
embedding_s= 512 # size of the embedding space
n_heads= 4 # n. of attention heads
ln_rate = 5e-4 # learning rate


# Select correct device
if torch.cuda.is_available():
    dev='cuda'
# Avoid 'mps' found indexing bug!
#elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
#    dev='mps'
else:
    dev='cpu'

# Generate some artificial data
x, target_x, start_token, end_token = generate_randomData_batch(max_seq_len=max_seq_len, dev=dev, vocab_size=vocab_size)

# set-up logger
setup_logger()


# Initialise overall stransformer architecture
transformer = Transformer(input_vocab_s=vocab_size, target_vocab_s=vocab_size, max_seq_len=max_seq_len, embedding_s=embedding_s, n_heads=n_heads, device=dev, ln_rate=ln_rate).to(dev)

# training:
for ep in range(tot_ep):
    # exclude last element of target seq since don't wanna predict next token for last element
    transformer_output = transformer(input_seq=x, target_seq=target_x[:,:-1])
    # shift target by one since want to predict next token
    loss = transformer.update(transformer_output.contiguous().view(-1, vocab_size), target_x[:,1:].contiguous().view(-1)) # target: [batchs_s*target_seq_len] since chosen loss takes (1d) class label
    logging.info(f" *** Episode: {ep} | Loss: {loss} *** ")

# Try inference by selecting sequence from the first batch 
# this is just to try the inference step since all inputs were random, 
# so we don't expect any meaningful learning beyond some memorisation
x = x[0:1,:]
target_x = target_x[0:1,:]
predictions = transformer.inference(input_seq=x, start_token=start_token, end_token=end_token, max_decoder_steps=max_seq_len)

## Check predicted and target sequences:
target = [t.cpu().item() for t in target_x.squeeze()]
logging.info(f" *** Predicted sequence: {predictions}")
logging.info(f" *** Target sequence: {target}")
