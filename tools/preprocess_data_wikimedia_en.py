import os
import json
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Define paths and parameters
dataset_name = "wikimedia/wikipedia"
subset_name = "20231102.en"
output_prefix = "./tokenized_wikipedia"
vocab_file = "./gpt2-vocab.json"
merge_file = "./gpt2-merges.txt"
tokenizer_type = "GPT2BPETokenizer"
append_eod = True
workers = 64

# Load the dataset
print("Loading dataset...")
dataset = load_dataset(dataset_name, subset_name, split='train')

# Initialize the tokenizer
print("Initializing tokenizer...")
tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merge_file)

# Function to tokenize a single document
def tokenize_document(document):
    tokens = tokenizer.encode(document['text'])
    if append_eod:
        tokens.append(tokenizer.eos_token_id)
    return tokens

# Tokenize the dataset
print("Tokenizing dataset...")
tokenized_data = [tokenize_document(doc) for doc in dataset]

# Save tokenized data to binary and index files
print("Saving tokenized data...")
output_bin_file = f"{output_prefix}_text_document.bin"
output_idx_file = f"{output_prefix}_text_document.idx"

# Save the tokenized data to a binary file
with open(output_bin_file, 'wb') as bin_file:
    for tokens in tokenized_data:
        np.array(tokens, dtype=np.int32).tofile(bin_file)

# Save the index data to a separate file
with open(output_idx_file, 'w') as idx_file:
    offset = 0
    for tokens in tokenized_data:
        length = len(tokens)
        idx_file.write(f"{offset}\t{length}\n")
        offset += length

print("Tokenized data saved successfully.")