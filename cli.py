#!/usr/bin/env python3
import argparse
import os
import json
import torch
from transformers import AutoTokenizer
import glob
from model import MetadataExtractor, extract_metadata, METADATA_FIELDS, MAX_LEN, DEVICE

def load_model(model_dir):
    """Load the trained model and label encoders"""
    from sklearn.preprocessing import LabelEncoder
    import torch.nn as nn
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load label encoders
    label_encoders = {}
    num_labels_dict = {}
    
    for field in METADATA_FIELDS:
        encoder_path = os.path.join(model_dir, f"{field}_encoder.json")
        with open(encoder_path, "r") as f:
            encoder_data = json.load(f)
        
        le = LabelEncoder()
        le.classes_ = encoder_data["classes"]
        label_encoders[field] = le
        num_labels_dict[field] = len(le.classes_)
    
    # Initialize model
    model = MetadataExtractor("distilbert-base-uncased", num_labels_dict)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer, label_encoders

def process_file(filename, model, tokenizer, label_encoders, output_format="json"):
    """Process a single file and extract metadata"""
    # Get file basename
    base_filename = os.path.basename(filename)
    
    # Extract metadata
    metadata = extract_metadata(model, tokenizer, label_encoders, base_filename)
    
    # Filter out empty or UNKNOWN values
    filtered_metadata = {field: value for field, value in metadata.items() 
                         if value != "UNKNOWN" and value != ""}
    
    # Add original filename
    filtered_metadata["filename"] = base_filename
    
    # Output based on format
    if output_format == "json":
        return filtered_metadata
    elif output_format == "csv":
        headers = ["filename"] + METADATA_FIELDS
        values = [filtered_metadata.get(field, "") for field in headers]
        return ",".join(values)
    else:
        # Pretty format
        result = f"File: {base_filename}\n"
        for field, value in filtered_metadata.items():
            if field != "filename":
                result += f"  {field}: {value}\n"
        return result

def main():
    parser = argparse.ArgumentParser(description="Extract metadata from media filenames")
    parser.add_argument("path", help="Path to file or directory to process")
    parser.add_argument("--model-dir", default="model_output", help="Directory containing the model files")
    parser.add_argument("--output", choices=["json", "csv", "pretty"], default="pretty", 
                        help="Output format (default: pretty)")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--output-file", help="Write output to a file instead of stdout")
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, label_encoders = load_model(args.model_dir)
    
    # Collect files to process
    files_to_process = []
    if os.path.isfile(args.path):
        files_to_process.append(args.path)
    else:
        # It's a directory
        if args.recursive:
            for dirpath, _, filenames in os.walk(args.path):
                for f in filenames:
                    if not f.startswith('.'):  # Skip hidden files
                        files_to_process.append(os.path.join(dirpath, f))
        else:
            files_to_process = [os.path.join(args.path, f) for f in os.listdir(args.path) 
                               if os.path.isfile(os.path.join(args.path, f)) and not f.startswith('.')]
    
    # Process files
    results = []
    for file in files_to_process:
        result = process_file(file, model, tokenizer, label_encoders, args.output)
        results.append(result)
    
    # Output results
    if args.output == "json":
        output = json.dumps(results, indent=2)
    elif args.output == "csv":
        headers = ["filename"] + METADATA_FIELDS
        output = ",".join(headers) + "\n" + "\n".join(results)
    else:
        output = "\n".join(results)
    
    # Write to file or stdout
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
        print(f"Results written to {args.output_file}")
    else:
        print(output)

if __name__ == "__main__":
    main()