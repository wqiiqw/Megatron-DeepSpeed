# Prepare dataset 
import os
import json
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer
from megatron.data import indexed_dataset

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

# Create indexed dataset builder
builder = indexed_dataset.make_builder(output_bin_file, impl='mmap', vocab_size=tokenizer.vocab_size)

# Add documents to the builder
for tokens in tokenized_data:
    builder.add_item(np.array(tokens, dtype=np.int32))

# Finalize the builder
builder.finalize(output_idx_file)

print("Tokenized data saved successfully.")
