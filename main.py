from multi30k_PreProcessing import Multi30k_PreProcessing
import torch
from transformer import Transformer
from utils import setup_logger
import logging

## Following PyTorch tutorial at: https://pytorch.org/tutorials/beginner/translation_transformer.html
## which uses the Multi30K dataset available from torch text datasets to train a German to English translation model
# Multi30K constist several (30k?) pairs of german and english setences.

# Select correct device
if torch.cuda.is_available():
    dev='cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
    dev='mps'
else:
    dev='cpu'

# set-up logger
setup_logger()

# Initalise Multi30k preprocessing class
multi30k_preProc = Multi30k_PreProcessing()

# Initialise useful variables
src_language = multi30k_preProc.SRC_LANGUAGE
tgt_language = multi30k_preProc.TGT_LANGUAGE
tot_ep = 100 # n. of training step
embedding_s= 512 # size of the embedding space
n_heads= 4 # n. of attention heads
ln_rate = 5e-4 # learning rate
src_vocab_s = len(multi30k_preProc.vocab_transform[src_language])
tgt_vocab_s = len(multi30k_preProc.vocab_transform[tgt_language])

print(src_vocab_s)
print(tgt_vocab_s)

# Initialise overall stransformer architecture

