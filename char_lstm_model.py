#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import json
import re
from typing import Dict, List, Tuple, Any, Optional
import string

# Configuration
CHAR_VOCAB = string.printable  # All printable ASCII characters
MAX_LEN = 200  # Maximum filename length in characters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Check for GPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Metal) device for GPU acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device for GPU acceleration")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device")

# Metadata fields we want to extract
METADATA_FIELDS = ["title", "year", "season", "episode", "episode_title"]

class CharacterLabeledDataset(Dataset):
    """Dataset for character-level sequence labeling"""
    
    def __init__(self, filenames, labels=None, char_to_idx=None, tag_to_idx=None):
        self.filenames = filenames
        self.labels = labels
        
        # Create character vocabulary mapping if not provided
        if char_to_idx is None:
            self.char_to_idx = {char: idx+1 for idx, char in enumerate(CHAR_VOCAB)}
            self.char_to_idx['<pad>'] = 0  # Padding token
            self.char_to_idx['<unk>'] = len(self.char_to_idx)  # Unknown token
        else:
            self.char_to_idx = char_to_idx
            
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Create tag vocabulary mapping if not provided
        if tag_to_idx is None:
            # Start with O tag for "Outside any entity"
            self.tag_to_idx = {'O': 0}
            # Add BIO tags for each metadata field
            for field in METADATA_FIELDS:
                self.tag_to_idx[f'B-{field}'] = len(self.tag_to_idx)  # Beginning of entity
                self.tag_to_idx[f'I-{field}'] = len(self.tag_to_idx)  # Inside entity
        else:
            self.tag_to_idx = tag_to_idx
            
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Convert filename to character indices
        char_indices = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in filename[:MAX_LEN]]
        
        # Pad if necessary
        if len(char_indices) < MAX_LEN:
            char_indices += [self.char_to_idx['<pad>']] * (MAX_LEN - len(char_indices))
        
        if self.labels is not None:
            # Get the tag sequence for this filename
            tag_sequence = self.labels[idx]
            
            # Convert tag sequence to indices
            tag_indices = [self.tag_to_idx.get(t, 0) for t in tag_sequence[:MAX_LEN]]
            
            # Pad if necessary
            if len(tag_indices) < MAX_LEN:
                tag_indices += [self.tag_to_idx['O']] * (MAX_LEN - len(tag_indices))
            
            return {
                'char_indices': torch.tensor(char_indices, dtype=torch.long),
                'tag_indices': torch.tensor(tag_indices, dtype=torch.long),
                'filename': filename
            }
        else:
            return {
                'char_indices': torch.tensor(char_indices, dtype=torch.long),
                'filename': filename
            }

class CharLSTM(nn.Module):
    """Character-level LSTM for sequence labeling"""
    
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(CharLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tag_size)  # *2 for bidirectional
    
    def forward(self, char_indices):
        """Forward pass through the network"""
        embedded = self.embedding(char_indices)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        tag_scores = self.hidden2tag(lstm_out)
        return tag_scores

def create_character_labels(filenames: List[str], metadata_dict: Dict[str, List[Any]]) -> List[List[str]]:
    """
    Create character-level BIO tags for each filename
    Returns a list of tag sequences, one for each filename
    """
    all_tag_sequences = []
    # Disable debug logging as it's too verbose
    debug_counter = 0  # Set to 0 to disable logging
    
    for idx, filename in enumerate(filenames):
        # Initialize all characters as "Outside" (O tag)
        tag_sequence = ['O'] * len(filename)
        
        # For debugging, print a few examples (currently disabled)
        if debug_counter > 0 and debug_counter < 5:  # Will never be true with debug_counter = 0
            print(f"\nCreating labels for: {filename}")
            print(f"Metadata: {[f'{k}: {metadata_dict[k][idx]}' for k in METADATA_FIELDS if metadata_dict[k][idx]]}")
            debug_counter += 1
        
        # For each metadata field, find its position in the filename
        for field in METADATA_FIELDS:
            value = str(metadata_dict[field][idx])
            
            # Skip empty values
            if not value or value.lower() in ('nan', ''):
                continue
            
            # Special handling for numeric fields like year
            if field == 'year' and value.isdigit() and len(value) == 4:
                # Look for exact year match
                year_pos = filename.find(value)
                if year_pos >= 0:
                    # Mark beginning and inside positions
                    tag_sequence[year_pos] = f'B-{field}'
                    for i in range(1, len(value)):
                        tag_sequence[year_pos + i] = f'I-{field}'
            
            # Special handling for season/episode patterns
            elif field in ['season', 'episode'] and str(value).replace('.', '', 1).isdigit():
                num_value = int(float(value))
                found = False
                
                # Common patterns for seasons: S01, Season 1, etc.
                if field == 'season':
                    patterns = [
                        f'S{num_value:02d}', f'S{num_value}',
                        f'Season{num_value}', f'Season {num_value}',
                        f's{num_value:02d}', f's{num_value}'
                    ]
                # Common patterns for episodes: E01, Episode 1, etc.
                else:
                    patterns = [
                        f'E{num_value:02d}', f'E{num_value}',
                        f'Episode{num_value}', f'Episode {num_value}',
                        f'e{num_value:02d}', f'e{num_value}',
                        f'- {num_value} ['  # Common pattern in anime filenames
                    ]
                
                # Check each pattern
                for pattern in patterns:
                    pattern_pos = filename.lower().find(pattern.lower())
                    if pattern_pos >= 0:
                        # Mark beginning position
                        tag_sequence[pattern_pos] = f'B-{field}'
                        # Mark inside positions
                        for i in range(1, len(pattern)):
                            if pattern_pos + i < len(tag_sequence):
                                tag_sequence[pattern_pos + i] = f'I-{field}'
                        found = True
                        break  # Stop after finding first match
                
                # Handle special case for anime-style episode numbers
                if not found and field == 'episode':
                    # Look for patterns like " - 01" or " - 1 "
                    pattern_pos = filename.find(f" - {num_value}")
                    if pattern_pos >= 0:
                        # Mark the number as episode
                        start_pos = pattern_pos + 3  # Skip " - "
                        tag_sequence[start_pos] = f'B-{field}'
                        # If it's a two-digit number
                        if num_value >= 10:
                            tag_sequence[start_pos + 1] = f'I-{field}'
            
            # For title fields, need special handling
            elif field == 'title':
                # Try different title extraction strategies
                
                # Strategy 1: For TV shows with dots (Breaking.Bad)
                if '.' in filename and 'S' in filename and 'E' in filename:
                    # Try to find title before season marker
                    s_pos = filename.find('S')
                    if s_pos > 0 and filename[s_pos+1:s_pos+3].isdigit():
                        title_part = filename[:s_pos].replace('.', ' ').strip()
                        if title_part:
                            for i, char in enumerate(title_part):
                                if i == 0:
                                    tag_sequence[i] = f'B-{field}'
                                else:
                                    tag_sequence[i] = f'I-{field}'
                
                # Strategy 2: For anime with brackets [Group] Title - 01
                elif '[' in filename and ']' in filename and ' - ' in filename:
                    bracket_end = filename.find(']')
                    dash_pos = filename.find(' - ')
                    
                    if bracket_end > 0 and dash_pos > bracket_end:
                        title_part = filename[bracket_end+1:dash_pos].strip()
                        start_pos = bracket_end + 1
                        
                        while start_pos < dash_pos and filename[start_pos] in ' \t':
                            start_pos += 1
                        
                        if start_pos < dash_pos:
                            # Mark title
                            for i in range(dash_pos - start_pos):
                                if i == 0:
                                    tag_sequence[start_pos] = f'B-{field}'
                                else:
                                    tag_sequence[start_pos + i] = f'I-{field}'
                
                # Strategy 3: For movies with year (Title.Year)
                elif field == 'title' and 'year' in metadata_dict and metadata_dict['year'][idx]:
                    year_val = str(metadata_dict['year'][idx])
                    if year_val.isdigit() and len(year_val) == 4:
                        year_pos = filename.find(year_val)
                        if year_pos > 0:
                            # Title is everything before the year
                            title_part = filename[:year_pos].rstrip('. -_')
                            if title_part:
                                for i, char in enumerate(title_part):
                                    if i == 0:
                                        tag_sequence[i] = f'B-{field}'
                                    else:
                                        tag_sequence[i] = f'I-{field}'
                
                # Try exact match as last resort
                if value and not any(t.startswith(f'B-{field}') for t in tag_sequence):
                    norm_value = value.lower()
                    norm_filename = filename.lower()
                    
                    if norm_value in norm_filename:
                        value_pos = norm_filename.find(norm_value)
                        # Mark beginning position
                        tag_sequence[value_pos] = f'B-{field}'
                        # Mark inside positions
                        for i in range(1, len(norm_value)):
                            tag_sequence[value_pos + i] = f'I-{field}'
            
            # For episode titles, try after the episode number
            elif field == 'episode_title' and value:
                # Look for episode title after season/episode markers
                se_pattern = re.search(r'S\d+E\d+|s\d+e\d+|\d+x\d+', filename)
                if se_pattern:
                    end_pos = se_pattern.end()
                    # The episode title often follows after a dot or space
                    if end_pos < len(filename) - 1:
                        # Skip separators
                        while end_pos < len(filename) and filename[end_pos] in '.-_ ':
                            end_pos += 1
                        
                        # Look for next separator (like period before resolution)
                        next_sep = -1
                        for sep in ['.', ' [', ' (', ' -']:
                            pos = filename[end_pos:].find(sep)
                            if pos != -1:
                                if next_sep == -1 or pos < next_sep:
                                    next_sep = pos
                        
                        if next_sep != -1:
                            title_end = end_pos + next_sep
                            # Mark episode title
                            for i in range(end_pos, title_end):
                                if i == end_pos:
                                    tag_sequence[i] = f'B-{field}'
                                else:
                                    tag_sequence[i] = f'I-{field}'
                
                # Exact match as fallback
                if value and not any(t.startswith(f'B-{field}') for t in tag_sequence):
                    norm_value = value.lower()
                    norm_filename = filename.lower()
                    
                    if norm_value in norm_filename:
                        value_pos = norm_filename.find(norm_value)
                        # Mark beginning position
                        tag_sequence[value_pos] = f'B-{field}'
                        # Mark inside positions
                        for i in range(1, len(norm_value)):
                            tag_sequence[value_pos + i] = f'I-{field}'
            
            # For all other string fields, do a flexible search
            else:
                # Normalize the value for better matching
                norm_value = value.lower()
                norm_filename = filename.lower()
                
                # Try to find exact matches
                if norm_value in norm_filename:
                    value_pos = norm_filename.find(norm_value)
                    # Mark beginning position
                    tag_sequence[value_pos] = f'B-{field}'
                    # Mark inside positions
                    for i in range(1, len(norm_value)):
                        tag_sequence[value_pos + i] = f'I-{field}'
        
        # For debugging, print a few labeled sequences (currently disabled)
        if debug_counter > 0 and debug_counter <= 5:  # Will never be true with debug_counter = 0
            print("Tag sequence:")
            non_o_tags = [(i, tag, filename[i]) for i, tag in enumerate(tag_sequence) if tag != 'O']
            for i, tag, char in non_o_tags:
                print(f"  {i}: {tag} ({char})")
            debug_counter -= 1
        
        all_tag_sequences.append(tag_sequence)
    
    return all_tag_sequences

def train_model(model, train_loader, val_loader, num_epochs=EPOCHS):
    """Train the character-level LSTM model"""
    print(f"Training model for {num_epochs} epochs...")
    
    # Define loss function and optimizer
    # We'll use CrossEntropyLoss for multi-class classification at each character position
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Track best model
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            char_indices = batch['char_indices'].to(DEVICE)
            tag_indices = batch['tag_indices'].to(DEVICE)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            tag_scores = model(char_indices)
            
            # Reshape for loss calculation
            # (batch_size, seq_len, num_tags) -> (batch_size * seq_len, num_tags)
            tag_scores = tag_scores.view(-1, tag_scores.shape[2])
            tag_indices = tag_indices.view(-1)
            
            # Calculate loss
            loss = criterion(tag_scores, tag_indices)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Calculate average loss
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                char_indices = batch['char_indices'].to(DEVICE)
                tag_indices = batch['tag_indices'].to(DEVICE)
                
                # Forward pass
                tag_scores = model(char_indices)
                
                # Reshape for loss calculation
                tag_scores = tag_scores.view(-1, tag_scores.shape[2])
                tag_indices = tag_indices.view(-1)
                
                # Calculate loss
                loss = criterion(tag_scores, tag_indices)
                
                total_val_loss += loss.item()
        
        # Calculate average loss
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def extract_metadata_from_tags(filename: str, tag_sequence: List[str]) -> Dict[str, str]:
    """
    Extract metadata values from a character-level tag sequence
    """
    metadata = {}
    
    # Track current field and value being built
    current_field = None
    current_value = []
    
    # Process each character and its tag
    for char, tag in zip(filename, tag_sequence):
        # Beginning of a new field
        if tag.startswith('B-'):
            # Save previous field if any
            if current_field:
                metadata[current_field] = ''.join(current_value)
                current_value = []
            
            # Start new field
            current_field = tag[2:]  # Remove 'B-' prefix
            current_value = [char]
        
        # Inside of current field
        elif tag.startswith('I-') and current_field and tag[2:] == current_field:
            current_value.append(char)
        
        # Outside any field
        elif tag == 'O':
            # Save previous field if any
            if current_field:
                metadata[current_field] = ''.join(current_value)
                current_field = None
                current_value = []
    
    # Save the last field if any
    if current_field:
        metadata[current_field] = ''.join(current_value)
    
    # Post-process extracted fields
    if metadata:
        # Clean up title (often contains separators)
        if 'title' in metadata:
            metadata['title'] = metadata['title'].replace('.', ' ').replace('_', ' ').strip()
        
        # Clean up episode title
        if 'episode_title' in metadata:
            metadata['episode_title'] = metadata['episode_title'].replace('.', ' ').replace('_', ' ').strip()
        
        # Format season consistently (S01, S1, etc.)
        if 'season' in metadata:
            season_value = metadata['season'].strip()
            # If it's just a number
            if season_value.isdigit():
                metadata['season'] = f"S{int(season_value)}"
            # If it starts with S or s
            elif season_value.lower().startswith('s') and season_value[1:].isdigit():
                metadata['season'] = f"S{int(season_value[1:])}"
        
        # Format episode consistently (E01, E1, etc.)
        if 'episode' in metadata:
            episode_value = metadata['episode'].strip()
            # If it's just a number
            if episode_value.isdigit():
                metadata['episode'] = f"E{int(episode_value)}"
            # If it starts with E or e
            elif episode_value.lower().startswith('e') and episode_value[1:].isdigit():
                metadata['episode'] = f"E{int(episode_value[1:])}"
    
    # Debug print if metadata is empty or incomplete - simplified version
    if len(metadata) < 2:
        # Just print a short warning
        print(f"Warning: Limited metadata extracted ({len(metadata)} fields)")
    
    return metadata

def extract_metadata(model, char_to_idx, idx_to_tag, filename):
    """
    Extract metadata from a filename using the trained model
    """
    model.eval()
    
    # Convert filename to character indices
    char_indices = [char_to_idx.get(c, char_to_idx['<unk>']) for c in filename[:MAX_LEN]]
    
    # Pad if necessary
    if len(char_indices) < MAX_LEN:
        char_indices += [char_to_idx['<pad>']] * (MAX_LEN - len(char_indices))
    
    # Convert to tensor and add batch dimension
    char_tensor = torch.tensor(char_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # Get predictions
    with torch.no_grad():
        tag_scores = model(char_tensor)
        tag_ids = torch.argmax(tag_scores, dim=2)[0].cpu().numpy()
    
    # Convert tag IDs to tag strings
    tag_sequence = [idx_to_tag[tag_id] for tag_id in tag_ids[:len(filename)]]
    
    # Extract metadata from tag sequence
    metadata = extract_metadata_from_tags(filename, tag_sequence)
    
    return metadata

def read_examples(filename: str) -> List[str]:
    """Read examples from a text file, skipping comments and empty lines"""
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                examples.append(line)
    return examples

def prepare_synthetic_data(num_examples=5000):
    """
    Generate synthetic data for training
    For now, we'll just load examples from a file
    In a real implementation, this would generate more complex examples
    """
    from generate_synthetic_data import generate_dataset, write_to_csv
    
    print(f"Generating {num_examples} synthetic examples...")
    
    # Generate synthetic dataset
    dataset = generate_dataset(num_examples)
    
    # Write to CSV
    csv_path = "synthetic_media_data.csv"
    write_to_csv(dataset, csv_path)
    
    print(f"Synthetic data written to {csv_path}")
    
    # Read real-world examples if available
    try:
        real_examples = read_examples("realistic_examples.txt")
        print(f"Read {len(real_examples)} real-world examples")
        
        # We could add these to the dataset, but for now let's use them for testing
        return csv_path, real_examples
    except FileNotFoundError:
        print("No real-world examples found, using synthetic data only")
        return csv_path, []

def main():
    """Main function to train and evaluate the model"""
    # Prepare data
    csv_path, real_examples = prepare_synthetic_data(5000)
    
    # Load data from CSV
    df = pd.read_csv(csv_path)
    df = df.fillna("")  # Replace NaN with empty strings
    
    # Make sure all required fields exist
    for field in METADATA_FIELDS:
        if field not in df.columns:
            df[field] = ""
    
    # Split data
    filenames = df["filename"].tolist()
    metadata_dict = {field: df[field].tolist() for field in METADATA_FIELDS}
    
    # Create character-level labels
    tag_sequences = create_character_labels(filenames, metadata_dict)
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        filenames, 
        tag_sequences,
        test_size=0.2, 
        random_state=42
    )
    
    # Create datasets
    train_dataset = CharacterLabeledDataset(X_train, y_train)
    val_dataset = CharacterLabeledDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    vocab_size = len(train_dataset.char_to_idx)
    tag_size = len(train_dataset.tag_to_idx)
    model = CharLSTM(
        vocab_size=vocab_size,
        tag_size=tag_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model.to(DEVICE)
    
    # Print model info
    print(f"Character vocabulary size: {vocab_size}")
    print(f"Tag vocabulary size: {tag_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    model = train_model(model, train_loader, val_loader)
    
    # Save model and vocabularies
    os.makedirs("char_model_output", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': train_dataset.char_to_idx,
        'tag_to_idx': train_dataset.tag_to_idx,
        'idx_to_tag': train_dataset.idx_to_tag,
        'config': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        }
    }, "char_model_output/model.pt")
    print("Model saved to char_model_output/model.pt")
    
    # Test on synthetic examples
    print("\nTesting model on synthetic examples:")
    synthetic_examples = [
        "Inception.2010.1080p.BluRay.x264-GROUP.mkv",
        "[HorribleSubs] My Hero Academia - 01 [1080p].mkv",
        "Breaking.Bad.S05E14.Ozymandias.1080p.BluRay.x264-GROUP.mkv"
    ]
    
    # Function to analyze and display results
    def evaluate_extraction(filename, expected=None):
        print(f"\nFilename: {filename}")
        
        # Get raw character-level predictions
        char_indices = [train_dataset.char_to_idx.get(c, train_dataset.char_to_idx['<unk>']) for c in filename[:MAX_LEN]]
        
        # Pad if necessary
        if len(char_indices) < MAX_LEN:
            char_indices += [train_dataset.char_to_idx['<pad>']] * (MAX_LEN - len(char_indices))
        
        # Convert to tensor and add batch dimension
        char_tensor = torch.tensor(char_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        # Get predictions
        with torch.no_grad():
            tag_scores = model(char_tensor)
            tag_ids = torch.argmax(tag_scores, dim=2)[0].cpu().numpy()
        
        # Convert tag IDs to tag strings for the actual characters in the filename
        tag_sequence = [train_dataset.idx_to_tag[tag_id] for tag_id in tag_ids[:len(filename)]]
        
        # Show a sample of raw tags (just the non-O ones)
        non_o_tags = [(i, tag, filename[i]) for i, tag in enumerate(tag_sequence) if tag != 'O']
        if non_o_tags:
            print("Raw tag examples:")
            for i, tag, char in non_o_tags[:10]:  # Show first 10
                print(f"  {i}: {tag} ({char})")
        else:
            print("No tags found! Model returned all 'O' tags.")
        
        # Extract and show metadata
        metadata = extract_metadata_from_tags(filename, tag_sequence)
        if metadata:
            print("Extracted metadata:")
            for field, value in metadata.items():
                print(f"  {field}: {value}")
        else:
            print("No metadata extracted!")
            
        # Compare to expected values if provided
        if expected:
            print("Expected values:")
            for field, value in expected.items():
                match = field in metadata and metadata[field] == value
                print(f"  {field}: {value} {'✓' if match else '✗'}")
    
    # Test each example
    for filename in synthetic_examples:
        evaluate_extraction(filename)
    
    # Test on real examples if available
    if real_examples:
        print("\n" + "="*50)
        print("Testing model on real-world examples:")
        
        # Test specific real examples with expected values
        real_test_cases = [
            {
                "filename": "[SubsPlease] Mushoku Tensei - 12 (1080p) [A5EC8478].mkv",
                "expected": {
                    "title": "Mushoku Tensei",
                    "episode": "E12"
                }
            },
            {
                "filename": "Breaking.Bad.S05E14.Ozymandias.1080p.BluRay.x264-GROUP.mkv",
                "expected": {
                    "title": "Breaking Bad",
                    "season": "S5",
                    "episode": "E14",
                    "episode_title": "Ozymandias"
                }
            },
            {
                "filename": "Inception.2010.1080p.BluRay.x264-GROUP.mkv",
                "expected": {
                    "title": "Inception",
                    "year": "2010"
                }
            }
        ]
        
        # Test specific examples
        for test_case in real_test_cases:
            evaluate_extraction(test_case["filename"], test_case["expected"])
        
        # Test remaining examples
        test_examples = [ex for ex in real_examples[:5] if ex not in [tc["filename"] for tc in real_test_cases]]
        for filename in test_examples:
            evaluate_extraction(filename)

if __name__ == "__main__":
    main()