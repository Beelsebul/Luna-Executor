import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

def get_tokenizer(file_path):
    # Creating a tokenizer with BPE(Byte pair encoding: https://en.wikipedia.org/wiki/Byte_pair_encoding) model
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    
    # Setting pre-tokenizer and decoder
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    
    # Adding special tokens
    special_tokens = ['<|startoftext|>', '<|endoftext|>', '<|unk|>', '<|pad|>']
    trainer = trainers.BpeTrainer(special_tokens=special_tokens)
    
    # Read data
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Train tokenizer
    tokenizer.train_from_iterator(lines, trainer=trainer)
    
    # Convert to PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    # Set special tokens. 
    # Although the tokens are already defined and tokenizer worked without this step, ChatGPT recommended to assign special tokens to the specific attribute.
    fast_tokenizer.pad_token = '<|pad|>'
    fast_tokenizer.bos_token = '<|startoftext|>'
    fast_tokenizer.eos_token = '<|endoftext|>'
    fast_tokenizer.unk_token = '<|unk|>'
    
    return fast_tokenizer

# Training tokenizer and saving it to the output folder
base_dir = os.path.join(os.path.dirname(__file__), '..')
data_dir = os.path.join(base_dir, 'gen_0_red_wins.txt')
tokenizer_dir = os.path.join(base_dir, 'tokenizer')

tokenizer = get_tokenizer(data_dir)
tokenizer.save_pretrained(tokenizer_dir)
