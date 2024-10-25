import torch
import torch.nn as nn
import torch.optim as opt
from embedding import Embedding, PositionalEmbedding
from encoder import EncoderBlock
from decoder import DecoderBlock

class Transformer(nn.Module):
    """ Transformer class to tie everything together"""

    def __init__(self, input_vocab_s, target_vocab_s, max_seq_len, embedding_s, n_heads, device, ln_rate=1e-3, n_blocks=2):
        """ 
            Args:
                input_vocab_s: vocabolary size of input sequence
                target_vocab_s: vocabolary size of target sequence
                embedding_s: input embedding size,
                n_heads: n. of attention heads
                n_blocks: n. of encoder and decoder layers
        """

        super().__init__()

        self.dev = device

        # Initialise embedding layers for encoder and decoder + positional embedding
        self.encoder_embedding = Embedding(vocab_size=input_vocab_s, embedding_s=embedding_s)
        self.decoder_embedding = Embedding(vocab_size=target_vocab_s, embedding_s=embedding_s)
        # Initialise positional embedding
        self.pos_embedding = PositionalEmbedding(embedding_s=embedding_s, max_seq_len=max_seq_len)

        # Initialise multiple encoder and decoder blocks
        self.encoder = nn.ModuleList([EncoderBlock(embedding_s=embedding_s, n_heads=n_heads) for _ in range(n_blocks)])
        self.decoder = nn.ModuleList([DecoderBlock(embedding_s=embedding_s, n_heads=n_heads) for _ in range(n_blocks)])
        
        # Initialise final linear layer
        self.l1 = nn.Linear(embedding_s, target_vocab_s)

        # Initialise optimiser
        self.optimizer = opt.Adam(self.parameters(), lr=ln_rate)

        # Initialise loss type
        self.criterion = nn.CrossEntropyLoss()

    def make_target_mask(self, target_seq):
        """ 
        Generate a mask for the target sequence 
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """

        batch_s = target_seq.shape[0]
        seq_s = target_seq.shape[1]

        # Note: the attention score has size [batch_s, seq_s, seq_s]
        return torch.tril(torch.ones((1, seq_s, seq_s))).to(self.dev)

    def forward(self, input_seq, target_seq):
        """ 
        Define forward pass for the entire transformer
        Args:
            input_seq: raw input sequence (e.g., word), [batch_s, seq_len]
            target_seq: raw output sequence (e.g., word), [batch_s, target_seq_len]
        """

        # input seq embedding
        input_emb = self.encoder_embedding(input_seq) #[batch_s, seq_len, embedding_s]
        input_emb = self.pos_embedding(input_emb) #[batch_s, seq_len, embedding_s]

        # Pass forward through each ecoder block with a loop
        encoder_output = input_emb
        for enc_block in self.encoder:
            encoder_output = enc_block(encoder_output) # NOTE: need to implement input seq mask for masking padding tokens!!!
        
        # Generate mask for the target sequence
        target_mask = self.make_target_mask(target_seq)

        # output seq embedding
        target_emb = self.decoder_embedding(target_seq) #[batch_s, target_seq_len, embedding_s]
        target_emb = self.pos_embedding(target_emb) #[batch_s, target_seq_len, embedding_s]

        # Pass forward through each decoder block with a loop
        for dec_block in self.decoder:
            target_emb = dec_block(x=target_emb, encoder_output=encoder_output, mask=target_mask) #[batch_s, target_seq_len, embedding_s]
        
        # Apply final linear layer (which can then be 'softmaxed' to get token probabilities - e.g., during inference)
        return self.l1(target_emb) # [batch_s, target_seq_len, vocab_s]

    def inference(self, input_seq, start_token, end_token, max_decoder_steps=100):
        """ 
        Perform the inference step, where given a single input sequence, the corresponding output 
        sequence is predicted in an auto-regressive way
        Args:
            input_seq: raw input sequence (e.g., word) [seq_s, embedding_s]
            start_token: token use to identify start of setence
            end_token: token use to identify end of setence
            max_decoder_steps: max n. of elements predicted in target seq
        """

        # input seq embedding
        input_emb = self.encoder_embedding(input_seq)
        input_emb = self.pos_embedding(input_emb)

        # Loop around encoder blocks
        encoder_output = input_emb
        for enc_block in self.encoder:
            encoder_output = enc_block(encoder_output)
        
        # Pass start token for decoder first step and then loop around
        # decoder predictions in an auto-regressive way to predict next token
        # until end_toke reached or max step size
        past_predictions = []
        prediction = torch.tensor([start_token], device=self.dev)
        past_predictions.append(prediction)
        t = 0
        # KEY: the model prediction at each step MUST be appended to all previous
        # predictions, which're then all passed as input to make the next prediction, 
        # Transformers have no implicit memory, can't only pass previous prediction like RNNs
        # otherwise it makes the next prediction only based on the previous token! 
        for t in range(max_decoder_steps):
            emb_prediction = self.decoder_embedding(torch.tensor(past_predictions, device=self.dev).unsqueeze(0))
            emb_prediction = self.pos_embedding(emb_prediction)
            # Pass forward through each decoder block with a loop
            decoder_output = emb_prediction
            for dec_block in self.decoder:
                decoder_output = dec_block(x=decoder_output, encoder_output=encoder_output, mask=None)
            
            # Apply final linear layer
            logit_prediction = self.l1(decoder_output)

            #NOTE: do not really need to compute softmax since we are taking the max value not sampling
            prediction = nn.functional.softmax(logit_prediction.squeeze(), dim=-1)
            prediction = torch.argmax(prediction, keepdim=True, dim=-1) # Select most likely element 
            # Append last prediction to all previous ones to make next prediction
            # NOTE: this step is slightly confusing, since at each step we get a new prediction for all past tokens
            # this is because at each step we are passing all past tokens to make a new prediction, resulting in the Transformer 
            # giving us a prediction for all of them. However, we only want to append the latest prediction - i.e., the one for the current token
            past_predictions.append(prediction[-1]) # only take prediction for last token

            if prediction[-1].item() == end_token:
                break
        
        return [t.cpu().item() for t in past_predictions]

    def update(self, prediction, target_seq):
        """ 
        Update entire transformer end-to-end
        Args:
            prediction: the transformer ouput
            target_seq: raw output sequence (e.g., word)
        """
        self.optimizer.zero_grad()
        # note nn.CrosEntropyLoss takes class number as targets (not one-hot) and logits as input
        loss = self.criterion(prediction, target_seq)
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
