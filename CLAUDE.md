# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run character-level model: `python char_lstm_model.py`
- Run CRF+LSTM model: `python crf_lstm_model.py`
- Run simplified model: `python simplified_model.py`
- Generate synthetic training data: `python generate_synthetic_data.py`
- Debug CLI output: `python cli.py --debug "filename"`

## Code Guidelines
- Use PEP 8 style for Python code
- Document classes and functions with docstrings
- Use type hints for function parameters and return values
- Prefer explicit character-level processing over regex pattern matching
- Model architecture: Continue using hybrid approach (CNN + BiLSTM + CRF)
- Training data: Use synthetic data with real-world examples for testing
- Use semantic class names like `CharFeatureExtractor`, `BiLSTM_CRF`
- Don't use regex for pattern matching in metadata extraction
- Organize imports: standard library, third-party, then local modules

## Project Status and Next Steps
Our BiLSTM-CRF model for metadata extraction from filenames still has issues with specific edge cases including:

1. Title boundary detection (truncation of titles like "Pacific Rim 2 Uprising" → "Pacific Rim 2")
2. Quality term inclusion in titles ("Lucy 2014 HC HDRip" instead of just "Lucy")
3. Special character preservation (hyphens in "X-Men", apostrophes in "Marvel's")
4. Subtitles after separators ("Movie - Subtitle" patterns)

We've enhanced the training data generation with generic, structurally varied synthetic examples and added a debug mode to the CLI that shows character-by-character tagging. Next steps include:

1. Using the debug tool to identify exact boundary detection issues
2. Consider model architecture improvements (increased capacity, enhanced convolutions)
3. Examine why the model fails to recognize certain metadata patterns
4. Focus on the recurring pattern where titles are truncated

The tag visualization feature will help diagnose exactly where the model is making incorrect boundary decisions.