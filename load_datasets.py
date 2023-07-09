import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import csv

def load_and_preprocess_data():
    data = []

    for root, dirs, files in os.walk('datasets'):
        for file in files:
            if file.endswith('.clean'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    data += f.readlines()


    print(f"Number of lines in data: {len(data)}")  # Debugging line

    # Tokenizer function using torchtext
    tokenizer = get_tokenizer('basic_english')

    # Tokenize data
    tokenized_data = [tokenizer(line) for line in data]

    print(f"Number of lines in tokenized data: {len(tokenized_data)}")  # Debugging line

    # Create vocab
    vocab = build_vocab_from_iterator(iter(tokenized_data))

    print(f"Vocabulary size: {len(vocab)}")  # Debugging line

    # Define the text pipeline
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    label_pipeline = lambda x: int(x) - 1

    return tokenized_data, vocab, text_pipeline, label_pipeline
