from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore")




class Multi30k_PreProcessing():
    """
    In order to prepare the data to be passed as input batches to Transformer need apply a series of transformation to the raw data
    to do so a function for each transformation must be initialised, specifically:
    - 1st: Define a tokenizer function which transform strings to tokens
    - 2nd: Define a Numericalization function which uses the tokenizer function to transform tokens into integer idexes
    - 3rd: Define a 'tensor_transform' function which add BOS/EOS tokens and create tensor for sequences of indexes
    Once we have these 3 transformation functions, we can define another function which triggers them sequentially:
    - 4th: Define a 'sequential_transform' function which allows to apply functions defined in previous 3 steps sequentially
    NOTE: In these steps we are simply defining each transformation function, without applying them to the data yet
    - 5th: define collate function to club all transformations together, which can be passed to DataLoader to create data src & trg batches
    """

    def __init__(self):

        ## ------------------------------
        # We need to modify the URLs for the dataset since the links to the original multi30k dataset are broken
        # Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
        multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
        multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
        ## ------------------------------

        ## Define useful variables
        self.SRC_LANGUAGE = 'de'
        self.TGT_LANGUAGE = 'en'
        # Place-holder
        self.token_transform = {}
        self.vocab_transform = {}
        # define special tokens: UNK: unknown token, PAD: padding token, BOS: beginning (of setence) token, EOS: end (of setence) token
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3 # NOTE: I think in my Transformer implementation assume padding token to be 0, NEED TO CHECK !!!
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

        ## --------------- 1st step: Tokenization function ----------------
        # Initialise tokenize functions for each language by selecting 'spacy' as the tokenizer library for each lang
        self.token_transform[self.SRC_LANGUAGE] = get_tokenizer(tokenizer='spacy', language='de_core_news_sm') # get_tokenizer: Generate tokenizer function for a string sentence
        self.token_transform[self.TGT_LANGUAGE] = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
        ## ----------------------------------------

        ## -------------- 2nd step: Numericalization function ---------------
        ## Initialise numericalization function for each language using build_vocab_from_iterator API
        # Usign for loop to initialize it for each language
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            # Select (iterable) training data 
            train_iter = Multi30k(split='train', language_pair=(self.SRC_LANGUAGE, self.TGT_LANGUAGE)) # Multi30k is the actual dataset class
            #print(next(iter(train_iter)))

            # build_vocab_from_iterator : build a Vocab from an interator, which must yield list or iterators of tokens
            # min_freq : The minimum frequency needed to include a token in the vocabulary.
            self.vocab_transform[ln] = build_vocab_from_iterator(iterator=self._yield_tokens(train_iter, ln), min_freq=1, specials=special_symbols, special_first=True)
        ## ----------------------------------------                                                    
                                                    
        # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
        # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
          self.vocab_transform[ln].set_default_index(self.UNK_IDX)

        ## -------------- 3rd step: 'tensor_transform'function ---------------
        # function to add BOS/EOS and create tensor for seq inputs 

        # this step consist in definying the  _tenro_transform() helper functions  below

        ## -------------- 4th step: 'sequential_transform' function ---------------
        # Initialise a text_transform dict containing func to apply the 3 defined transformations for each lang
        self.text_transform = {}
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            self.text_transform[ln] = self._sequential_transforms(self.token_transform[ln], # Tokenization
                                                       self.vocab_transform[ln], # Numericalization
                                                       self._tensor_transform)    # Add BOS/EOS and create tensor


    ## ---------- 5th step: Create collate function which can be passed to DataLoader to create tensor batches appropriately -----
    # This function uses 'text_tranform' to apply all the transformations to a batch of raw data extracted by DataLoader
    def collate_fn(self, batch):
        src_batch, tgt_batch = [],[]
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.SRC_LANGUAGE](src_sample.rstrip("\n"))) # rstrip(char) removes specified char at the end of a string
            tgt_batch.append(self.text_transform[self.TGT_LANGUAGE](tgt_sample.rstrip("\n")))
        
        # pad_sequence: pad a list of variable length Tesnors with padding_value
        src_batch = pad_sequence(src_batch, padding_value= self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value= self.PAD_IDX)

        return src_batch, tgt_batch

## ----------- Define useful helper functions ---------------
    def _yield_tokens(self, data_iter, language):
        """ helper function to yield ('iterable return') list of tokens by using the pre-defined tokenizer (i.e., token_transform = spacy, in this case)"""
        lang_index = {self.SRC_LANGUAGE: 0, self.TGT_LANGUAGE: 1} 
        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[lang_index[language]]) # select correct tokenizer for lang and sample for that lang

    # Define function to add BOS/EoS and create tensor for seq of input indexes
    def _tensor_transform(self, token_ids):
        return torch.cat([torch.tensor([self.BOS_IDX]), torch.tensor([token_ids]), torch.tensor([self.EOS_IDX])])

    # Define function to perform sequential operations together
    def _sequential_transforms(self, *transformations):
        def func(txt_input):
            for t in transformations:
                txt_input = t(txt_input)
            return txt_input
        return func


