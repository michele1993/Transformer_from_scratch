from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
import warnings
warnings.filterwarnings("ignore")


## Following PyTorch tutorial at: https://pytorch.org/tutorials/beginner/translation_transformer.html
## which uses the Multi30K dataset available from torch text datasets to train a German to English translation model
# Multi30K constist several (30k?) pairs of german and english setences.

## In order to prepare the data to be passed as input batches to Transformer need apply a series of transformation to the raw data
## to do so a function for each transformation must be initialised, specifically:
# 1st: Define a tokenizer function which transform string to tokens
# 2nd: Define a Numericalization function which uses the tokenizer function to transform tokens to intger idexes
# 3rd: Define a 'tensor_transform' function which add BOS/EOS tokens and create tensor for sequences of indexes
# Once we have these 3 transformation functions, we can define another function which triggers them sequentially:
# 4th: Define a 'sequential_transform' function which allows to apply functions defined in previous 3 steps sequentially

## ------------------------------
# We need to modify the URLs for the dataset since the links to the original multi30k dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
## ------------------------------

## Define useful variables
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
# Place-holder
token_transform = {}
vocab_transform = {}
# define special tokens: UNK: unknown token, PAD: padding token, BOS: beginning (of setence) token, EOS: end (of setence) token
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3 # NOTE: I think in my Transformer implementation assume padding token to be 0, NEED TO CHECK !!!
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

## --------------- 1st step: Tokenization function ----------------
# Initialise tokenize functions for each language by selecting 'spacy' as the tokenizer library for each lang
token_transform[SRC_LANGUAGE] = get_tokenizer(tokenizer='spacy', language='de_core_news_sm') # get_tokenizer: Generate tokenizer function for a string sentence
token_transform[TGT_LANGUAGE] = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
## ----------------------------------------

## -------------- 2nd step: Numericalization function ---------------
## Initialise numericalization function for each language using build_vocab_from_iterator API

# define useful help function
def yield_tokens(data_iter, language):
    """ helper function to yield ('iterable return') list of tokens by using pre-defined tokenizer (i.e., token_transform = spacy)"""
    lang_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1} 
    for data_sample in data_iter:
        yield token_transform[language](data_sample[lang_index[language]])

# Use for loop to initialize Numericalization function for each language
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Select (iterable) training data 
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)) # Multi30k is the actual dataset class

    # build_vocab_from_iterator : build a Vocab from an interator, which must yield list or iterators of tokens
    # min_freq : The minimum frequency needed to include a token in the vocabulary.
    vocab_transform[ln] = build_vocab_from_iterator(iterator=yield_tokens(train_iter, ln), min_freq=1, specials=special_symbols, special_first=True)
## ----------------------------------------                                                    
                                                    
# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

## -------------- 3rd step: 'tensor_transform'function ---------------

## -------------------------------------------


## -------------- 4th step: 'sequential_transform' function ---------------

## -------------------------------------------
