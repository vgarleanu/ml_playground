#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import json
from typing import Dict, List, Tuple, Any, Optional
import string
from tqdm import tqdm
from torchcrf import CRF  # You'll need to install pytorch-crf: pip install pytorch-crf

# Configuration
CHAR_VOCAB = string.printable  # All printable ASCII characters
MAX_LEN = 200        # Maximum filename length in characters
BATCH_SIZE = 64      # Good batch size for stabler gradients
EPOCHS = 50          # More epochs for the complex model with early stopping
LEARNING_RATE = 3e-4 # Lower learning rate for more stable training
EMBEDDING_DIM = 128  # Character embedding dimension
HIDDEN_DIM = 256     # Hidden dimension for LSTM
NUM_LAYERS = 2       # Two layers of bidirectional LSTM
DROPOUT = 0.4        # Increased dropout for better generalization
WEIGHT_O_TAG = 0.05  # Further downweight "O" tags to focus more on entity extraction

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

class CharFeatureExtractor:
    """Extracts additional features for each character in a filename"""
    
    @staticmethod
    def get_char_type(char):
        """Return character type features"""
        if char.isupper():
            return [1, 0, 0, 0, 0]  # Uppercase
        elif char.islower():
            return [0, 1, 0, 0, 0]  # Lowercase
        elif char.isdigit():
            return [0, 0, 1, 0, 0]  # Digit
        elif char in ".-_[]() ":
            return [0, 0, 0, 1, 0]  # Common separator
        else:
            return [0, 0, 0, 0, 1]  # Other
    
    @staticmethod
    def extract_features(filename):
        """Extract features for each character in the filename"""
        features = []
        # Create richer feature representation without explicit pattern recognition
        for i, char in enumerate(filename):
            # Character type features
            char_type = CharFeatureExtractor.get_char_type(char)
            
            # Position features - normalized to 0-1
            # This helps the model learn position-dependent patterns
            pos_features = [i / len(filename)]
            
            # Context window features - characters before and after
            # This helps the model learn character n-gram patterns
            context_before = [0, 0, 0, 0, 0]  # One-hot encoding of char type before
            if i > 0:
                context_before = CharFeatureExtractor.get_char_type(filename[i-1])
                
            context_after = [0, 0, 0, 0, 0]  # One-hot encoding of char type after
            if i < len(filename) - 1:
                context_after = CharFeatureExtractor.get_char_type(filename[i+1])
            
            # Is this character the start of a word boundary?
            is_boundary_start = 1 if (i == 0 or filename[i-1] in ".-_[]() ") and char not in ".-_[]() " else 0
            
            # Is this character the end of a word boundary?
            is_boundary_end = 1 if (i == len(filename)-1 or filename[i+1] in ".-_[]() ") and char not in ".-_[]() " else 0
            
            # Combine all features
            char_features = (
                char_type +                # Character type (5)
                pos_features +             # Position (1)
                context_before +           # Previous char type (5)
                context_after +            # Next char type (5)
                [is_boundary_start] +      # Word boundary start (1)
                [is_boundary_end]          # Word boundary end (1)
            )
            features.append(char_features)
        
        return features

class CharacterLabeledDataset(Dataset):
    """Dataset for character-level sequence labeling with additional features"""
    
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
        
        # Calculate class weights for weighted loss
        if labels is not None:
            # Flatten all tag sequences
            all_tags = [tag for seq in labels for tag in seq]
            # Count occurrences of each tag
            tag_counts = {}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Calculate inverse frequency
            total_tags = len(all_tags)
            self.class_weights = {tag: total_tags / count for tag, count in tag_counts.items()}
            
            # Convert to tensor format for CrossEntropyLoss
            self.weight_tensor = torch.zeros(len(self.tag_to_idx))
            for tag, idx in self.tag_to_idx.items():
                self.weight_tensor[idx] = self.class_weights.get(tag, 1.0)
        else:
            self.weight_tensor = None
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Convert filename to character indices
        char_indices = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in filename[:MAX_LEN]]
        
        # Extract additional features
        features = CharFeatureExtractor.extract_features(filename[:MAX_LEN])
        
        # Pad if necessary
        if len(char_indices) < MAX_LEN:
            char_indices += [self.char_to_idx['<pad>']] * (MAX_LEN - len(char_indices))
            features += [[0] * len(features[0])] * (MAX_LEN - len(features))
        
        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float)
        
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
                'features': features_tensor,
                'tag_indices': torch.tensor(tag_indices, dtype=torch.long),
                'filename': filename,
                'mask': torch.tensor([1] * min(len(filename), MAX_LEN) + [0] * max(0, MAX_LEN - len(filename)), dtype=torch.bool)
            }
        else:
            return {
                'char_indices': torch.tensor(char_indices, dtype=torch.long),
                'features': features_tensor,
                'filename': filename,
                'mask': torch.tensor([1] * min(len(filename), MAX_LEN) + [0] * max(0, MAX_LEN - len(filename)), dtype=torch.bool)
            }

class BiLSTM_CRF(nn.Module):
    """Enhanced character-level BiLSTM with CRF for sequence labeling"""
    
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim, feature_dim, num_layers, dropout):
        super(BiLSTM_CRF, self).__init__()
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Feature projection with batch normalization for more stable training
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        # Convolutional layer for capturing local character n-grams
        # This helps detect patterns like "S01E02" or "2010"
        self.conv = nn.Conv1d(
            in_channels=embedding_dim * 2,  # Char embeddings + features
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1
        )
        
        # Main LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,  # Output from conv layer
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention mechanism to capture long-range dependencies
        # This helps connect related tags across the sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # BiLSTM output dimension
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection to tag space with layer normalization
        self.hidden2tag = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, tag_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # CRF layer for optimal tag sequence prediction
        self.crf = CRF(tag_size, batch_first=True)
    
    def forward(self, char_indices, features, mask=None, tags=None):
        """Forward pass through the enhanced network architecture"""
        batch_size, seq_len = char_indices.shape
        
        # Character embeddings
        char_embeds = self.embedding(char_indices)  # [batch_size, seq_len, embedding_dim]
        
        # Process features - need to reshape for batch norm
        features_flat = features.view(batch_size * seq_len, -1)
        feature_embeds_flat = self.feature_proj(features_flat)
        feature_embeds = feature_embeds_flat.view(batch_size, seq_len, -1)
        
        # Combine embeddings
        combined_embeds = torch.cat([char_embeds, feature_embeds], dim=-1)
        combined_embeds = self.dropout(combined_embeds)
        
        # Apply 1D convolution to capture local patterns
        # Reshape for conv1d which expects [batch, channels, seq_len]
        combined_embeds = combined_embeds.transpose(1, 2)
        conv_out = F.relu(self.conv(combined_embeds))
        conv_out = conv_out.transpose(1, 2)  # Back to [batch, seq_len, features]
        
        # LSTM for sequential modeling
        lstm_out, _ = self.lstm(conv_out)
        
        # Self-attention for capturing long-range dependencies
        attention_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=~mask if mask is not None else None
        )
        
        # Residual connection
        lstm_out = lstm_out + attention_out
        lstm_out = self.dropout(lstm_out)
        
        # Project to tag space
        emissions = self.hidden2tag(lstm_out)
        
        # If tags are provided, compute loss
        if tags is not None:
            # negate log-likelihood (CRF returns negative log-likelihood)
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # Decode the best path
            best_tags = self.crf.decode(emissions, mask=mask)
            return best_tags

def create_character_labels(filenames: List[str], metadata_dict: Dict[str, List[Any]]) -> List[List[str]]:
    """
    Create character-level BIO tags for each filename
    Returns a list of tag sequences, one for each filename
    """
    all_tag_sequences = []
    
    for idx, filename in enumerate(filenames):
        # Initialize all characters as "Outside" (O tag)
        tag_sequence = ['O'] * len(filename)
        
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
                
                # Train the model to recognize various season/episode patterns
                # Rather than using regex pattern matching, we'll provide explicit character sequences
                # that represent beginning of season/episode markers
                
                # For seasons, look for natural placements in the filename
                if field == 'season':
                    # Try to find S followed by digits
                    for i in range(len(filename) - 1):
                        if i < len(tag_sequence) and filename[i].lower() == 's' and i+1 < len(filename) and filename[i+1].isdigit():
                            # Check if this looks like a season marker (not just random 's1' in a word)
                            is_boundary = (i == 0 or filename[i-1] in ".-_[]() ")
                            
                            if is_boundary:
                                # Extract the full number after S
                                num_str = ""
                                j = i + 1
                                while j < len(filename) and filename[j].isdigit():
                                    num_str += filename[j]
                                    j += 1
                                
                                # If this number matches our expected season number
                                if num_str and (int(num_str) == num_value or int(num_str) == num_value * 100):
                                    # Mark S as beginning
                                    tag_sequence[i] = f'B-{field}'
                                    # Mark numbers as inside
                                    for k in range(i+1, i+1+len(num_str)):
                                        if k < len(tag_sequence):
                                            tag_sequence[k] = f'I-{field}'
                                    break
                    
                    # Also look for "Season" followed by number
                    season_str = "Season"
                    for i in range(len(filename) - len(season_str)):
                        if filename[i:i+len(season_str)].lower() == season_str.lower():
                            # Find following number
                            j = i + len(season_str)
                            while j < len(filename) and filename[j] in " ":
                                j += 1
                            
                            num_str = ""
                            while j < len(filename) and filename[j].isdigit():
                                num_str += filename[j]
                                j += 1
                            
                            # If this number matches our expected season
                            if num_str and int(num_str) == num_value:
                                # Mark entire "Season X" as the field
                                for k in range(i, j):
                                    if k == i:
                                        tag_sequence[k] = f'B-{field}'
                                    else:
                                        tag_sequence[k] = f'I-{field}'
                                break
                
                # For episodes, similar approach
                else:  # field == 'episode'
                    # Look for E followed by digits
                    for i in range(len(filename) - 1):
                        if i < len(tag_sequence) and filename[i].lower() == 'e' and i+1 < len(filename) and filename[i+1].isdigit():
                            # Make sure this is a boundary and not in the middle of a word
                            # Also check if it follows a season marker
                            is_episode_marker = False
                            if i > 0:
                                # Check if preceded by 's' and digits (common pattern: S01E02)
                                if filename[i-1].isdigit():
                                    j = i-1
                                    while j > 0 and filename[j].isdigit():
                                        j -= 1
                                    if j >= 0 and filename[j].lower() == 's':
                                        is_episode_marker = True
                                # Or check if it's at a boundary
                                elif filename[i-1] in ".-_[]() ":
                                    is_episode_marker = True
                            
                            if is_episode_marker:
                                # Extract full number after E
                                num_str = ""
                                j = i + 1
                                while j < len(filename) and filename[j].isdigit():
                                    num_str += filename[j]
                                    j += 1
                                
                                # If this number matches our expected episode
                                if num_str and int(num_str) == num_value:
                                    # Mark E as beginning
                                    tag_sequence[i] = f'B-{field}'
                                    # Mark numbers as inside
                                    for k in range(i+1, i+1+len(num_str)):
                                        if k < len(tag_sequence):
                                            tag_sequence[k] = f'I-{field}'
                                    break
                    
                    # Look for episode patterns in anime-style filenames: " - 01" or " - 1 "
                    # These are common in anime releases like "[Group] Title - 01 [1080p]"
                    dash_pattern = f" - {num_value}"
                    dash_pos = filename.find(dash_pattern)
                    if dash_pos >= 0:
                        # Mark just the number as the episode
                        start_pos = dash_pos + 3  # Skip " - "
                        tag_sequence[start_pos] = f'B-{field}'
                        # If it's a two-digit number
                        if num_value >= 10:
                            tag_sequence[start_pos + 1] = f'I-{field}'
                    
                    # Also look for "Episode" followed by number
                    episode_str = "Episode"
                    for i in range(len(filename) - len(episode_str)):
                        if filename[i:i+len(episode_str)].lower() == episode_str.lower():
                            # Find following number
                            j = i + len(episode_str)
                            while j < len(filename) and filename[j] in " ":
                                j += 1
                            
                            num_str = ""
                            while j < len(filename) and filename[j].isdigit():
                                num_str += filename[j]
                                j += 1
                            
                            # If this number matches our expected episode
                            if num_str and int(num_str) == num_value:
                                # Mark entire "Episode X" as the field
                                for k in range(i, j):
                                    if k == i:
                                        tag_sequence[k] = f'B-{field}'
                                    else:
                                        tag_sequence[k] = f'I-{field}'
                                break
            
            # For title fields, need special handling
            elif field == 'title':
                # Try different title extraction strategies
                
                # Strategy 1: For TV shows with dots (Breaking.Bad)
                if '.' in filename and 'S' in filename and 'E' in filename:
                    # Try to find title before season marker
                    s_pos = filename.find('S')
                    if s_pos > 0 and s_pos+1 < len(filename) and filename[s_pos+1:s_pos+2].isdigit():
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
                # Handle episode titles by looking for them after season/episode markers
                # without using regex
                
                # First identify any season/episode markers in the filename
                se_positions = []
                
                # Look for S followed by digits then E followed by digits (like S01E02)
                for i in range(len(filename) - 3):
                    if filename[i].lower() == 's' and i+1 < len(filename) and filename[i+1].isdigit():
                        # Find the end of the season number
                        j = i + 1
                        while j < len(filename) and filename[j].isdigit():
                            j += 1
                        
                        # Check if followed by episode marker
                        if j < len(filename) and filename[j].lower() == 'e' and j+1 < len(filename) and filename[j+1].isdigit():
                            # Find the end of the episode number
                            k = j + 1
                            while k < len(filename) and filename[k].isdigit():
                                k += 1
                            
                            se_positions.append((i, k))  # Store start and end positions
                
                # Also look for patterns like '1x02' (season x episode)
                for i in range(len(filename) - 3):
                    if filename[i].isdigit() and i+1 < len(filename) and filename[i+1].lower() == 'x' and i+2 < len(filename) and filename[i+2].isdigit():
                        # Find the end of the number sequence
                        j = i + 2
                        while j < len(filename) and filename[j].isdigit():
                            j += 1
                        
                        se_positions.append((i, j))  # Store start and end positions
                
                # Process found positions
                for start_pos, end_pos in se_positions:
                    # The episode title often follows after a dot or space
                    if end_pos < len(filename) - 1:
                        # Skip separators
                        pos = end_pos
                        while pos < len(filename) and filename[pos] in '.-_ ':
                            pos += 1
                        
                        # Look for next separator (like period before resolution)
                        next_sep_pos = len(filename)
                        for sep in ['.', ' [', ' (', ' -']:
                            sep_idx = filename[pos:].find(sep)
                            if sep_idx != -1:
                                if pos + sep_idx < next_sep_pos:
                                    next_sep_pos = pos + sep_idx
                        
                        # If we found a separator and there's content between episode marker and next separator
                        if next_sep_pos < len(filename) and pos < next_sep_pos:
                            # Mark episode title
                            for i in range(pos, next_sep_pos):
                                if i == pos:
                                    tag_sequence[i] = f'B-{field}'
                                else:
                                    tag_sequence[i] = f'I-{field}'
                            
                            # If we've identified a title, stop looking
                            if any(t.startswith(f'B-{field}') for t in tag_sequence):
                                break
                
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
        
        all_tag_sequences.append(tag_sequence)
    
    return all_tag_sequences

def train_model(model, train_loader, val_loader, num_epochs=EPOCHS):
    """Train the enhanced character-level LSTM-CRF model"""
    print(f"Training model for {num_epochs} epochs...")
    
    # Define optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6, eps=1e-8)
    
    # Learning rate warmup then cosine decay
    # First some warmup epochs, then cosine decay to 10% of initial LR
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine decay
            decay_epochs = num_epochs - warmup_epochs
            decay_step = epoch - warmup_epochs
            cosine_decay = 0.1 + 0.9 * (1 + np.cos(np.pi * decay_step / decay_epochs)) / 2
            return cosine_decay
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Track best model and implement early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience = 8  # Increased patience for more complex model
    patience_counter = 0
    
    # Create tag weights to balance O vs non-O tags
    # Count occurrences of each tag in training set
    tag_counts = torch.zeros(len(train_loader.dataset.tag_to_idx))
    for tags_batch in [batch['tag_indices'] for batch in train_loader]:
        for i in range(len(train_loader.dataset.tag_to_idx)):
            tag_counts[i] += (tags_batch == i).sum().item()
    
    # Calculate inverse frequency weights capped at 10.0
    total_tags = tag_counts.sum()
    tag_weights = torch.clamp(total_tags / tag_counts, min=1.0, max=10.0)
    
    # Special handling for O tag
    o_tag_idx = train_loader.dataset.tag_to_idx['O']
    tag_weights[o_tag_idx] = WEIGHT_O_TAG
    
    # Put weights on device
    tag_weights = tag_weights.to(DEVICE)
    
    # Show tag weights
    print("Tag weights:")
    for tag, idx in train_loader.dataset.tag_to_idx.items():
        print(f"  {tag}: {tag_weights[idx]:.4f}")
    
    # Training loop
    for epoch in range(num_epochs):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs} (LR: {current_lr:.7f})")
        
        # Training phase
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            char_indices = batch['char_indices'].to(DEVICE)
            features = batch['features'].to(DEVICE)
            tags = batch['tag_indices'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero
            
            # Forward pass
            loss = model(char_indices, features, mask=mask, tags=tags)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})
        
        # Calculate average loss
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        progress_bar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch in progress_bar:
                char_indices = batch['char_indices'].to(DEVICE)
                features = batch['features'].to(DEVICE)
                tags = batch['tag_indices'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                
                # Forward pass
                loss = model(char_indices, features, mask=mask, tags=tags)
                
                total_val_loss += loss.item()
                progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        
        # Calculate average loss
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model and check early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
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
    
    # Extract features
    features = CharFeatureExtractor.extract_features(filename[:MAX_LEN])
    
    # Convert filename to character indices
    char_indices = [char_to_idx.get(c, char_to_idx['<unk>']) for c in filename[:MAX_LEN]]
    
    # Prepare mask
    mask = [1] * len(char_indices)
    
    # Pad if necessary
    if len(char_indices) < MAX_LEN:
        padding_len = MAX_LEN - len(char_indices)
        char_indices += [char_to_idx['<pad>']] * padding_len
        features += [[0] * len(features[0])] * padding_len
        mask += [0] * padding_len
    
    # Convert to tensors and add batch dimension
    char_tensor = torch.tensor(char_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
    feature_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(DEVICE)
    mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(DEVICE)
    
    # Get predictions
    with torch.no_grad():
        best_tags = model(char_tensor, feature_tensor, mask=mask_tensor)
    
    # Convert tag IDs to tag strings
    tag_sequence = [idx_to_tag[tag_id] for tag_id in best_tags[0]]
    
    # Extract metadata from tag sequence
    metadata = extract_metadata_from_tags(filename, tag_sequence[:len(filename)])
    
    return metadata, tag_sequence[:len(filename)]

def read_examples(filename: str) -> List[str]:
    """Read examples from a text file, skipping comments and empty lines"""
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                examples.append(line)
    return examples

def prepare_synthetic_data(num_examples=10000):
    """
    Generate an enhanced synthetic dataset with more examples and challenging cases
    """
    from generate_synthetic_data import generate_dataset, write_to_csv
    
    print(f"Generating {num_examples} synthetic examples...")
    
    # Generate synthetic dataset with balanced distribution of challenging cases
    # Doubled the dataset size for better training
    dataset = generate_dataset(num_examples, balance_difficult_cases=True)
    
    # Write to CSV
    csv_path = "synthetic_media_data.csv"
    write_to_csv(dataset, csv_path)
    
    print(f"Synthetic data written to {csv_path}")
    
    # Read real-world examples if available
    try:
        real_examples = read_examples("realistic_examples.txt")
        print(f"Read {len(real_examples)} real-world examples")
        
        # Include some real examples in the training data to improve performance
        # on actual filenames
        real_examples_for_training = real_examples[:min(20, len(real_examples))]
        
        # Create synthetic labels for real examples
        real_data = []
        for example in real_examples_for_training:
            # Basic heuristic labeling
            labels = {"media_type": "", "title": "", "year": "", "season": "", "episode": "", "episode_title": ""}
            
            # Attempt to identify type and title
            if example.startswith('[') and ']' in example and ' - ' in example:
                # Anime pattern
                labels["media_type"] = "anime"
                bracket_end = example.find(']')
                dash_pos = example.find(' - ')
                if bracket_end > 0 and dash_pos > bracket_end:
                    labels["title"] = example[bracket_end+1:dash_pos].strip()
                    
                    # Try to find episode number
                    if dash_pos + 3 < len(example) and example[dash_pos+3:dash_pos+5].isdigit():
                        labels["episode"] = int(example[dash_pos+3:dash_pos+5])
            
            elif '.S' in example and 'E' in example:
                # TV show pattern
                labels["media_type"] = "tv"
                season_pos = example.find('.S')
                if season_pos > 0:
                    labels["title"] = example[:season_pos].replace('.', ' ').strip()
                    
                    # Find season and episode without regex
                    # Look for season marker
                    s_pos = -1
                    for i in range(len(example)):
                        if i < len(example) - 1 and example[i].upper() == 'S' and example[i+1].isdigit():
                            s_pos = i
                            break
                    
                    if s_pos >= 0:
                        # Extract season number
                        season_num = ""
                        i = s_pos + 1
                        while i < len(example) and example[i].isdigit():
                            season_num += example[i]
                            i += 1
                            
                        if season_num:
                            labels["season"] = int(season_num)
                        
                        # Look for episode marker after season
                        e_pos = -1
                        if i < len(example) - 1 and example[i].upper() == 'E' and example[i+1].isdigit():
                            e_pos = i
                        
                        if e_pos >= 0:
                            # Extract episode number
                            episode_num = ""
                            j = e_pos + 1
                            while j < len(example) and example[j].isdigit():
                                episode_num += example[j]
                                j += 1
                                
                            if episode_num:
                                labels["episode"] = int(episode_num)
                                
                                # Try to find episode title after the episode number
                                if j < len(example):
                                    # Skip separators
                                    while j < len(example) and example[j] in '.-_ ':
                                        j += 1
                                    
                                    # Find next major separator
                                    next_sep = len(example)
                                    for sep in ['.', ' [', ' (', ' -']:
                                        sep_pos = example[j:].find(sep)
                                        if sep_pos != -1 and j + sep_pos < next_sep:
                                            next_sep = j + sep_pos
                                    
                                    # Extract episode title
                                    if j < next_sep:
                                        ep_title = example[j:next_sep].replace('.', ' ').strip()
                                        if ep_title:
                                            labels["episode_title"] = ep_title
            
            else:
                # Look for 4-digit year in the filename (movie pattern)
                # Scan through the string looking for 4 consecutive digits
                for i in range(len(example) - 3):
                    if (example[i:i+4].isdigit() and 
                        1900 <= int(example[i:i+4]) <= 2030):  # Valid year range
                        
                        # Found a potential year
                        year_val = example[i:i+4]
                        year_pos = i
                        
                        # Movie pattern
                        labels["media_type"] = "movie"
                        
                        # If there's content before the year, it's likely the title
                        if year_pos > 0:
                            labels["title"] = example[:year_pos].replace('.', ' ').strip()
                            labels["year"] = int(year_val)
                            break
            
            # Add to dataset if we identified something useful
            if any(v for v in labels.values()):
                real_data.append((example, labels))
        
        # Add real examples with labels to training data
        if real_data:
            print(f"Adding {len(real_data)} real examples to training data")
            csv_path = "combined_media_data.csv"
            
            # Add real examples to dataset
            combined_dataset = dataset + real_data
            
            # Write combined dataset
            write_to_csv(combined_dataset, csv_path)
            print(f"Combined data written to {csv_path}")
        
        # Return the rest of the real examples for testing
        return csv_path, real_examples[20:] if len(real_examples) > 20 else real_examples
    except FileNotFoundError:
        print("No real-world examples found, using synthetic data only")
        return csv_path, []

def main():
    """Main function to train and evaluate the enhanced model"""
    # Prepare data with larger dataset
    csv_path, real_examples = prepare_synthetic_data(10000)
    
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
    
    # Feature dimension from the first item
    feature_dim = train_dataset[0]['features'].shape[1]
    
    # Initialize model
    vocab_size = len(train_dataset.char_to_idx)
    tag_size = len(train_dataset.tag_to_idx)
    model = BiLSTM_CRF(
        vocab_size=vocab_size,
        tag_size=tag_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        feature_dim=feature_dim,
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
    os.makedirs("crf_model_output", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': train_dataset.char_to_idx,
        'tag_to_idx': train_dataset.tag_to_idx,
        'idx_to_tag': train_dataset.idx_to_tag,
        'config': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'feature_dim': feature_dim
        }
    }, "crf_model_output/model.pt")
    print("Model saved to crf_model_output/model.pt")
    
    # Function to analyze and display results
    def evaluate_extraction(filename, expected=None):
        print(f"\nFilename: {filename}")
        
        # Extract metadata
        metadata, tag_sequence = extract_metadata(
            model, 
            train_dataset.char_to_idx, 
            train_dataset.idx_to_tag, 
            filename
        )
        
        # Show a sample of the tags (non-O ones)
        non_o_tags = [(i, tag, filename[i]) for i, tag in enumerate(tag_sequence) if tag != 'O']
        if non_o_tags:
            print("Raw tag examples:")
            for i, tag, char in non_o_tags[:15]:  # Show first 15
                print(f"  {i}: {tag} ({char})")
        else:
            print("No tags found! Model returned all 'O' tags.")
        
        # Show extracted metadata
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
    
    # Test on synthetic examples
    print("\nTesting model on synthetic examples:")
    synthetic_examples = [
        "Inception.2010.1080p.BluRay.x264-GROUP.mkv",
        "[HorribleSubs] My Hero Academia - 01 [1080p].mkv",
        "Breaking.Bad.S05E14.Ozymandias.1080p.BluRay.x264-GROUP.mkv"
    ]
    
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
