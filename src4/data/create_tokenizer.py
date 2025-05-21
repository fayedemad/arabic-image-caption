import os
from transformers import AutoTokenizer

def create_tokenizer(caption_files, output_dir, vocab_size=64000):
    """
    Create and train a new tokenizer on Arabic captions.
    
    Args:
        caption_files: List of paths to caption files
        output_dir: Directory to save the tokenizer
        vocab_size: Size of the vocabulary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Reading captions...")
    captions = []
    for caption_file in caption_files:
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                caption = line.strip().split('\t')[1]  # Get caption part
                captions.append(caption)
    
    print(f"Read {len(captions)} captions")
    
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    
    print("Training tokenizer...")
    tokenizer = tokenizer.train_new_from_iterator(
        captions,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True
    )
    
    print(f"Saving tokenizer to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    
    print("\nTokenizer Statistics:")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    print("\nTesting tokenizer on a sample caption:")
    sample_caption = captions[0]
    print(f"Original: {sample_caption}")
    tokens = tokenizer.tokenize(sample_caption)
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    return tokenizer

if __name__ == '__main__':
    create_tokenizer(
        caption_files=[
            'data/train_captions.txt',
            'data/val_captions.txt'
        ],
        output_dir='tokenizer_new',
        vocab_size=9462
    ) 