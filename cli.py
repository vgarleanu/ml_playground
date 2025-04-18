#!/usr/bin/env python3
import argparse
import os
import json
import sys
import torch
from crf_lstm_model import (
    BiLSTM_CRF, CharFeatureExtractor, extract_metadata, METADATA_FIELDS, 
    MAX_LEN, DEVICE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
)

def load_crf_model(model_path="crf_model_output/model.pt"):
    """Load the trained CRF-LSTM model"""
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load saved model and config
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Extract model configuration
    config = checkpoint.get('config', {
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'feature_dim': 18  # Default value, will be overridden from checkpoint
    })
    
    # Get character and tag mappings
    char_to_idx = checkpoint['char_to_idx']
    tag_to_idx = checkpoint['tag_to_idx']
    idx_to_tag = checkpoint['idx_to_tag']
    
    # Initialize model with the same architecture
    model = BiLSTM_CRF(
        vocab_size=len(char_to_idx),
        tag_size=len(tag_to_idx),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        feature_dim=config['feature_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model, char_to_idx, idx_to_tag

def process_filename(filename, model, char_to_idx, idx_to_tag, output_format="pretty"):
    """Process a single filename and extract metadata"""
    # Extract metadata using the CRF-LSTM model
    metadata, tag_sequence = extract_metadata(model, char_to_idx, idx_to_tag, filename)
    
    # Format output
    if output_format == "json":
        # Add filename to the output
        result = metadata.copy()
        result["filename"] = filename
        return result
    elif output_format == "csv":
        headers = ["filename"] + METADATA_FIELDS
        values = [filename] + [metadata.get(field, "") for field in METADATA_FIELDS]
        return ",".join(str(v) for v in values)
    else:
        # Pretty format
        result = f"Filename: {filename}\n"
        if metadata:
            for field, value in metadata.items():
                result += f"  {field}: {value}\n"
        else:
            result += "  No metadata extracted\n"
        return result

def main():
    parser = argparse.ArgumentParser(description="Extract metadata from media filenames using CRF-LSTM model")
    parser.add_argument("--model", default="crf_model_output/model.pt", 
                        help="Path to the trained model file (default: crf_model_output/model.pt)")
    parser.add_argument("--output", choices=["json", "csv", "pretty"], default="pretty", 
                        help="Output format (default: pretty)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode, reading from stdin")
    parser.add_argument("filenames", nargs="*", help="Filenames to process (not used in interactive mode)")
    args = parser.parse_args()
    
    # Load model
    model, char_to_idx, idx_to_tag = load_crf_model(args.model)
    
    if args.interactive:
        # Interactive mode: read filenames from stdin
        print("Enter filenames (one per line, Ctrl+D to exit):")
        try:
            for line in sys.stdin:
                filename = line.strip()
                if filename:
                    result = process_filename(filename, model, char_to_idx, idx_to_tag, args.output)
                    if args.output == "json":
                        print(json.dumps(result, indent=2))
                    else:
                        print(result)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        # Batch mode: process all provided filenames
        if not args.filenames:
            parser.print_help()
            sys.exit(1)
        
        results = []
        for filename in args.filenames:
            result = process_filename(os.path.basename(filename), model, char_to_idx, idx_to_tag, args.output)
            results.append(result)
        
        # Output results based on format
        if args.output == "json":
            print(json.dumps(results, indent=2))
        elif args.output == "csv":
            headers = ["filename"] + METADATA_FIELDS
            print(",".join(headers))
            for result in results:
                print(result)
        else:
            for result in results:
                print(result)

if __name__ == "__main__":
    main()