import os
import json
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define paths and parameters
dataset_name = "wikimedia/wikipedia"
subset_name = "20231101.en"
output_prefix = "./tokenized_wikipedia"
vocab_file = "./gpt2-vocab.json"
merge_file = "./gpt2-merges.txt"
tokenizer_type = "GPT2BPETokenizer"
append_eod = True
workers = 64
NUM_OF_TEXT_ITEMS = 0 # 1000  # Set this to the desired number of text items to process, 0 means process all

# Load the dataset
print("Loading dataset...")
dataset = load_dataset(dataset_name, subset_name, split='train')

# Sort the dataset by a consistent key (e.g., document ID or text)
#dataset = sorted(dataset, key=lambda x: x['text'])

# If NUM_OF_TEXT_ITEMS is specified and non-zero, limit the dataset
if NUM_OF_TEXT_ITEMS > 0:
    dataset = dataset[:NUM_OF_TEXT_ITEMS]

# Initialize the tokenizer
print("Initializing tokenizer...")
tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merge_file)

# Function to tokenize a single document
def tokenize_document(document):
    tokens = tokenizer.encode(document['text'])
    if append_eod:
        tokens.append(tokenizer.eos_token_id)
    return tokens

# Tokenize the dataset using multiple worker threads with a progress bar
print("Tokenizing dataset...")
tokenized_data = []

with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(tokenize_document, doc) for doc in dataset]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing"):
        tokenized_data.append(future.result())

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
