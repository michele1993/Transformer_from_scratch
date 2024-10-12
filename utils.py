import logging
import torch

def setup_logger(seed=None):
    """ set useful logger set-up"""
    logging.basicConfig(format='%(asctime)s %(message)s', encoding='utf-8', level=logging.INFO)
    logging.debug(f'Pytorch version: {torch.__version__}')
    if seed is not None:
        logging.info(f'Seed: {seed}')

def generate_randomData_batch(max_seq_len, dev, batch_s=50, input_seq_s=10, target_seq_s=12, vocab_size=50):


    assert target_seq_s <= max_seq_len and input_seq_s <= max_seq_len, "Sequence size larger than max sequence size"

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

    return x, target_x, start_token, end_token

