#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
import os
import json
import re
from typing import Dict, List, Tuple

# Configuration
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 3e-5
MODEL_NAME = "distilbert-base-uncased"  # Lightweight model
# Check for MPS (Metal Performance Shaders) for Mac GPUs
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Metal) device for GPU acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device for GPU acceleration")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device")

# Simplified metadata fields - focusing on the most critical ones
METADATA_FIELDS = ["title", "year", "season", "episode", "episode_title"]

# Tag space for token classification (BIO scheme)
TAGS = ["O"]  # O = Outside any entity
for field in METADATA_FIELDS:
    TAGS.append(f"B-{field}")  # Beginning of entity
    TAGS.append(f"I-{field}")  # Inside entity

class TokenClassificationDataset(Dataset):
    def __init__(self, filenames, tokenized_texts=None, tag_sequences=None, tokenizer=None, max_len=MAX_LEN):
        self.filenames = filenames
        self.tokenized_texts = tokenized_texts
        self.tag_sequences = tag_sequences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = {tag: idx for idx, tag in enumerate(TAGS)}
        self.id2tag = {idx: tag for idx, tag in enumerate(TAGS)}
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        if self.tokenized_texts is not None and self.tag_sequences is not None:
            # Use pre-tokenized texts and tags
            tokenized_text = self.tokenized_texts[idx]
            tag_sequence = self.tag_sequences[idx]
            
            # Convert tags to IDs
            tag_ids = [self.tag2id[tag] for tag in tag_sequence]
            
            # Pad or truncate
            if len(tokenized_text) > self.max_len - 2:  # -2 for [CLS] and [SEP]
                tokenized_text = tokenized_text[:self.max_len - 2]
                tag_ids = tag_ids[:self.max_len - 2]
            
            # Add special tokens
            input_ids = [self.tokenizer.cls_token_id] + tokenized_text + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            
            # Add tags for special tokens (use "O" tag)
            tag_ids = [self.tag2id["O"]] + tag_ids + [self.tag2id["O"]]
            
            # Padding
            padding_length = self.max_len - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                tag_ids = tag_ids + ([self.tag2id["O"]] * padding_length)
            
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(tag_ids, dtype=torch.long)
            }
        
        # For inference only (no labels)
        encoding = self.tokenizer(
            filename,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

def normalize_filename(filename):
    """Pre-process filename to help with token classification"""
    # Very minimal normalization - just replace common separators with spaces
    # This maintains the structure while making it easier for tokenization
    filename = re.sub(r'[._]', ' ', filename)
    
    # Keep hyphens when they're part of a word, but separate them when they're separators
    filename = re.sub(r'(\w)-(\w)', r'\1_HYPHEN_\2', filename)  # Replace hyphens in words with temp placeholder
    filename = re.sub(r'-', ' ', filename)                       # Replace separator hyphens with spaces
    filename = re.sub(r'_HYPHEN_', '-', filename)                # Restore hyphens in words
    
    # Make sure brackets have spaces around them to be properly tokenized
    filename = re.sub(r'\[', ' [ ', filename)
    filename = re.sub(r'\]', ' ] ', filename)
    
    # Remove duplicate spaces
    filename = re.sub(r'\s+', ' ', filename).strip()
    
    return filename

def tokenize_and_align_labels(filenames, metadata_dict, tokenizer):
    """
    Tokenize filenames and create BIO tag sequences
    Uses a simpler, more robust approach to find metadata in filenames
    """
    tokenized_texts = []
    tag_sequences = []
    
    for idx, filename in enumerate(filenames):
        # Normalize and tokenize the filename
        normalized_filename = normalize_filename(filename)
        tokens = tokenizer.tokenize(normalized_filename)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Initialize all tags as "O" (outside)
        tags = ["O"] * len(tokens)
        
        # Prepare a helper function to find and tag sequences
        def find_and_tag_sequence(search_value, field_name):
            if not search_value or str(search_value).lower() in ('nan', ''):
                return False
            
            # Convert to string and normalize the same way as the filename
            search_value = str(search_value)
            normalized_search = normalize_filename(search_value).lower()
            
            # Skip very short values (likely noise)
            if len(normalized_search) < 2:
                return False
                
            # Tokenize the search value
            search_tokens = tokenizer.tokenize(normalized_search)
            
            # If search value is a single token, we need exact matching
            if len(search_tokens) == 1:
                for i, token in enumerate(tokens):
                    if token.lower() == search_tokens[0].lower():
                        tags[i] = f"B-{field_name}"
                        return True
                        
            # For multi-token search values, we need to look for the sequence
            else:
                normalized_filename_lower = normalized_filename.lower()
                
                # First check if the value appears in the filename at all
                if normalized_search in normalized_filename_lower:
                    # Scan through tokens looking for matching sequences
                    for i in range(len(tokens) - len(search_tokens) + 1):
                        match_found = True
                        
                        # Check if tokens at position i match search_tokens
                        for j in range(len(search_tokens)):
                            # Handle wordpiece tokens (##)
                            t1 = tokens[i+j].lower().replace('##', '')
                            t2 = search_tokens[j].lower().replace('##', '')
                            
                            if t1 != t2:
                                match_found = False
                                break
                                
                        if match_found:
                            # Tag the sequence
                            tags[i] = f"B-{field_name}"
                            for j in range(1, len(search_tokens)):
                                if i+j < len(tags):
                                    tags[i+j] = f"I-{field_name}"
                            return True
            
            # More flexible matching for titles and episode titles
            if field_name in ['title', 'episode_title']:
                # We'll count how many tokens from search_tokens are found in sequence
                best_match_start = -1
                best_match_length = 0
                
                for i in range(len(tokens) - 1):  # Need at least 2 tokens
                    match_length = 0
                    for j in range(min(len(search_tokens), len(tokens) - i)):
                        t1 = tokens[i+j].lower().replace('##', '')
                        t2 = search_tokens[j].lower().replace('##', '')
                        
                        # More flexible matching for titles
                        if t1 == t2 or (len(t1) > 2 and len(t2) > 2 and (t1 in t2 or t2 in t1)):
                            match_length += 1
                        else:
                            break
                    
                    # Update best match if we found a better one
                    if match_length > best_match_length and match_length >= 2:
                        best_match_length = match_length
                        best_match_start = i
                
                # If we found a decent match, tag it
                if best_match_start >= 0 and best_match_length >= 2:
                    tags[best_match_start] = f"B-{field_name}"
                    for j in range(1, best_match_length):
                        if best_match_start + j < len(tags):
                            tags[best_match_start + j] = f"I-{field_name}"
                    return True
            
            # Special handling for season/episode numbers
            if field_name in ['season', 'episode'] and str(search_value).replace('.', '', 1).isdigit():
                # Convert to integer
                num_value = int(float(search_value))
                
                # Look for season/episode patterns in the normalized filename
                for i, token in enumerate(tokens):
                    token_lower = token.lower()
                    
                    # Check for season patterns like 'S1', 'S01', 'Season 1'
                    if field_name == 'season':
                        if (token_lower.startswith('s') and token_lower[1:].isdigit() and int(token_lower[1:]) == num_value) or \
                           (token_lower == 'season' and i+1 < len(tokens) and tokens[i+1].isdigit() and int(tokens[i+1]) == num_value):
                            tags[i] = f"B-{field_name}"
                            # Tag the next token if it's part of "Season 1" pattern
                            if token_lower == 'season' and i+1 < len(tokens):
                                tags[i+1] = f"I-{field_name}"
                            return True
                    
                    # Check for episode patterns like 'E1', 'E01', 'Episode 1'
                    if field_name == 'episode':
                        if (token_lower.startswith('e') and token_lower[1:].isdigit() and int(token_lower[1:]) == num_value) or \
                           (token_lower == 'episode' and i+1 < len(tokens) and tokens[i+1].isdigit() and int(tokens[i+1]) == num_value) or \
                           (i > 0 and tokens[i-1].lower() == '-' and token.isdigit() and int(token) == num_value):  # Anime pattern: "- 01"
                            tags[i] = f"B-{field_name}"
                            # Tag the next token if it's part of "Episode 1" pattern
                            if token_lower == 'episode' and i+1 < len(tokens):
                                tags[i+1] = f"I-{field_name}"
                            return True
            
            # Special handling for year (4-digit number)
            if field_name == 'year' and search_value.isdigit() and len(search_value) == 4:
                for i, token in enumerate(tokens):
                    if token.isdigit() and len(token) == 4 and token == search_value:
                        tags[i] = f"B-{field_name}"
                        return True
            
            return False
        
        # Process each field in order of importance
        for field in METADATA_FIELDS:
            find_and_tag_sequence(metadata_dict[field][idx], field)
        
        tokenized_texts.append(token_ids)
        tag_sequences.append(tags)
    
    return tokenized_texts, tag_sequences

def prepare_data(csv_path):
    """Prepare data for training from the CSV file"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Replace NaN with empty strings
    df = df.fillna("")
    
    # Make sure all metadata fields exist in dataframe
    for field in METADATA_FIELDS:
        if field not in df.columns:
            df[field] = ""
    
    # Split data
    filenames = df["filename"].tolist()
    metadata_dict = {field: df[field].tolist() for field in METADATA_FIELDS}
    
    X_train, X_val, indices_train, indices_val = train_test_split(
        filenames, 
        list(range(len(filenames))),
        test_size=0.2, 
        random_state=42
    )
    
    # Create metadata dictionaries for train and val sets
    y_train_dict = {field: [metadata_dict[field][i] for i in indices_train] for field in METADATA_FIELDS}
    y_val_dict = {field: [metadata_dict[field][i] for i in indices_val] for field in METADATA_FIELDS}
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train_dict, y_val_dict

def train_model(model, train_dataloader, val_dataloader, num_epochs=EPOCHS):
    """Train the model"""
    print(f"Training model for {num_epochs} epochs...")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # Track best model
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # Forward pass
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        # Calculate average loss
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Training loss: {avg_train_loss}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
        
        # Calculate average loss
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation loss: {best_val_loss}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def save_model(model, tokenizer, save_dir="simplified_model_output"):
    """Save the model, tokenizer, and label encoders"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save tag mapping
    tag_mapping = {
        "tags": TAGS,
        "tag2id": {tag: idx for idx, tag in enumerate(TAGS)},
        "id2tag": {str(idx): tag for idx, tag in enumerate(TAGS)}
    }
    
    with open(os.path.join(save_dir, "tag_mapping.json"), "w") as f:
        json.dump(tag_mapping, f)
    
    print(f"Model saved to {save_dir}")

def extract_metadata(model, tokenizer, filename):
    """Extract metadata from a filename using the trained model"""
    model.eval()
    
    # Normalize and tokenize the filename
    normalized_filename = normalize_filename(filename)
    
    # Tokenize the filename
    encoding = tokenizer(
        normalized_filename,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get tag IDs
    tag_predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    
    # Get tokens from input IDs
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Load tag mapping
    id2tag = {idx: tag for idx, tag in enumerate(TAGS)}
    
    # Extract metadata
    metadata = {}
    current_field = None
    current_value = []
    
    for i, (token, tag_id) in enumerate(zip(tokens, tag_predictions)):
        tag = id2tag[tag_id]
        
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        
        # Skip "O" tags
        if tag == "O":
            if current_field:
                # Save the current value
                field_name = current_field
                field_value = " ".join(current_value)
                
                # Clean up field value
                field_value = field_value.replace(" ##", "").replace("##", "")
                field_value = re.sub(r'\s+', ' ', field_value).strip()
                
                # Store the value
                if field_name in metadata:
                    # If field exists, append to it
                    metadata[field_name] += " " + field_value
                else:
                    metadata[field_name] = field_value
                
                current_field = None
                current_value = []
            continue
        
        # Check if this is a beginning tag
        if tag.startswith("B-"):
            # If we were collecting a previous field, save it
            if current_field:
                field_name = current_field
                field_value = " ".join(current_value)
                
                # Clean up field value
                field_value = field_value.replace(" ##", "").replace("##", "")
                field_value = re.sub(r'\s+', ' ', field_value).strip()
                
                # Store the value
                if field_name in metadata:
                    # If field exists, append to it
                    metadata[field_name] += " " + field_value
                else:
                    metadata[field_name] = field_value
            
            # Start a new field
            current_field = tag[2:]  # Remove "B-" prefix
            current_value = [token]
        
        # Inside tag - continue current field
        elif tag.startswith("I-") and current_field and current_field == tag[2:]:
            current_value.append(token)
    
    # Save the last field if any
    if current_field:
        field_name = current_field
        field_value = " ".join(current_value)
        
        # Clean up field value
        field_value = field_value.replace(" ##", "").replace("##", "")
        field_value = re.sub(r'\s+', ' ', field_value).strip()
        
        # Store the value
        if field_name in metadata:
            # If field exists, append to it
            metadata[field_name] += " " + field_value
        else:
            metadata[field_name] = field_value
    
    # Post-process metadata
    for field, value in metadata.items():
        # Clean up any remaining special tokens or formatting
        value = value.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")
        value = re.sub(r'\s+', ' ', value).strip()
        metadata[field] = value
    
    return metadata

def main():
    """Main function to train and save the model"""
    print(f"Using device: {DEVICE}")
    
    # Load data
    X_train, X_val, y_train_dict, y_val_dict = prepare_data("synthetic_media_data.csv")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize and create token classification dataset
    print("Tokenizing and aligning labels...")
    train_tokens, train_tags = tokenize_and_align_labels(X_train, y_train_dict, tokenizer)
    val_tokens, val_tags = tokenize_and_align_labels(X_val, y_val_dict, tokenizer)
    
    # Initialize datasets
    train_dataset = TokenClassificationDataset(
        X_train, train_tokens, train_tags, tokenizer
    )
    val_dataset = TokenClassificationDataset(
        X_val, val_tokens, val_tags, tokenizer
    )
    
    # Initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    num_labels = len(TAGS)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    model.to(DEVICE)
    
    # Define the id2tag and tag2id dictionaries
    id2tag = {idx: tag for idx, tag in enumerate(TAGS)}
    tag2id = {tag: idx for idx, tag in enumerate(TAGS)}
    
    # Set the id2tag and tag2id dictionaries in the model config
    model.config.id2label = id2tag
    model.config.label2id = tag2id
    
    # Train model
    model = train_model(model, train_dataloader, val_dataloader)
    
    # Save model
    save_model(model, tokenizer)
    
    # Test model on a few examples
    test_filenames = [
        "Inception.2010.1080p.BluRay.x264-GROUP.mkv",
        "[HorribleSubs] My Hero Academia - 01 [1080p].mkv",
        "Breaking.Bad.S05E14.Ozymandias.1080p.BluRay.x264-GROUP.mkv"
    ]
    
    print("\nTesting model on examples:")
    for filename in test_filenames:
        print(f"\nFilename: {filename}")
        metadata = extract_metadata(model, tokenizer, filename)
        for field, value in metadata.items():
            if value:
                print(f"  {field}: {value}")

if __name__ == "__main__":
    main()