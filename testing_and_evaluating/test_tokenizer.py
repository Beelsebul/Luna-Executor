import os
from transformers import PreTrainedTokenizerFast

def test_new_tokenizer(tokenizer_path):
    # Loading the new tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Test text
    test_text = "A11 A10 K10"
    
    print("Original Text:\n", test_text)
    
    # Tokenizing the text
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    print("\nTokens and their IDs:")
    for token, token_id in zip(tokens, token_ids):
        print(f"{token}: {token_id}")

    # Displaying the tokenizer's vocabulary
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    
    print("\nFull Vocabulary:")
    for token, token_id in sorted_vocab:
        print(f"{token}: {token_id}")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    tok_dir = os.path.join(base_dir, 'tokenizer')
    test_new_tokenizer(tok_dir)
