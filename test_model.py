import torch
from model import LyricsGenerator
from data_preprocessing import TextPreprocessor
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
import json
from torch.utils.data import DataLoader, TensorDataset

try:
    nltk.download('punkt')
except:
    pass

def load_and_preprocess_data():
    """Load and preprocess the data with proper encoding"""
    try:
        # Load datasets with explicit encoding
        spotify_df = pd.read_csv('Spotify Million Song Dataset_exported.csv', encoding='latin1')
        top500_df = pd.read_csv('Top 500 Songs.csv', encoding='latin1')
        
        # Extract lyrics/text
        spotify_lyrics = spotify_df['text'].dropna().tolist()  # Spotify dataset uses 'text'
        top500_lyrics = top500_df['description'].dropna().tolist()  # Top 500 uses 'description'
        
        # Combine lyrics
        all_lyrics = spotify_lyrics + top500_lyrics
        
        # Basic cleaning
        all_lyrics = [str(text) for text in all_lyrics if isinstance(text, str)]
        
        print(f"Loaded {len(spotify_lyrics)} Spotify lyrics and {len(top500_lyrics)} Top 500 lyrics")
        return all_lyrics
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return []

def create_test_batch(lyrics, preprocessor, seq_length=50):
    """Create a test batch from lyrics"""
    sequences = []
    for text in lyrics:
        # Convert text to sequence
        sequence = []
        words = preprocessor.clean_text(text).split()
        for word in words:
            if word in preprocessor.word2idx:
                sequence.append(preprocessor.word2idx[word])
            else:
                sequence.append(preprocessor.word2idx['<UNK>'])
        
        # Pad sequence
        if len(sequence) > seq_length:
            sequence = sequence[:seq_length]
        else:
            sequence = sequence + [preprocessor.word2idx['<PAD>']] * (seq_length - len(sequence))
        
        sequences.append(sequence)
    
    return torch.tensor(sequences)

def test_model_basic(model, preprocessor, device):
    """Basic model testing - generate samples with different settings"""
    print("\nBasic Model Testing:")
    
    test_prompts = [
        "I love",
        "The music",
        "Dancing in",
        "My heart"
    ]
    
    temperatures = [0.5, 1.0, 1.5]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            
            # Convert prompt to sequence
            words = prompt.split()
            sequence = [preprocessor.word2idx.get(word, preprocessor.word2idx['<UNK>']) 
                       for word in words]
            sequence = [preprocessor.word2idx['<START>']] + sequence  # Add START token
            current_seq = torch.tensor([sequence], device=device).unsqueeze(0)
            
            # Generate
            model.eval()
            with torch.no_grad():
                generated = []
                hidden = None
                
                for _ in range(50):  # Generate up to 50 words
                    output, hidden = model(current_seq.squeeze(0), hidden)
                    output = output[:, -1, :] / temp
                    probs = torch.softmax(output, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    if next_token.item() == preprocessor.word2idx['<END>']:
                        break
                        
                    # Use get() with <UNK> as default for unknown indices
                    next_word = preprocessor.idx2word.get(str(next_token.item()), '<UNK>')
                    if next_word == '<UNK>':
                        continue
                        
                    generated.append(next_word)
                    current_seq = next_token.unsqueeze(0)
            
            print(' '.join(generated))

def test_model_diversity(model, preprocessor, device, n_samples=5):
    """Test model's ability to generate diverse outputs"""
    print("\nDiversity Testing:")
    
    prompt = "The"
    temperature = 1.0
    samples = []
    
    # Convert prompt to sequence
    sequence = [preprocessor.word2idx['<START>']]  # Start with START token
    sequence += [preprocessor.word2idx.get(word, preprocessor.word2idx['<UNK>']) 
                for word in prompt.split()]
    sequence = torch.tensor(sequence, device=device)  # Shape: [seq_len]
    
    # Generate multiple samples
    for i in range(n_samples):
        model.eval()
        with torch.no_grad():
            generated = []
            current_seq = sequence.clone()
            hidden = None
            
            for _ in range(50):
                # Forward pass
                output, hidden = model(current_seq, hidden)
                
                # Get next token probabilities
                if len(output.shape) == 3:
                    output = output[:, -1, :]  # Take last token if sequence
                output = output / temperature
                probs = torch.softmax(output, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs[0], 1)
                
                if next_token.item() == preprocessor.word2idx['<END>']:
                    break
                    
                # Use get() with <UNK> as default for unknown indices
                next_word = preprocessor.idx2word.get(str(next_token.item()), '<UNK>')
                if next_word == '<UNK>':
                    continue
                    
                generated.append(next_word)
                current_seq = next_token
        
        samples.append(' '.join(generated))
    
    # Print samples
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(sample)
    
    # Calculate diversity metrics
    unique_words = set()
    total_words = 0
    for sample in samples:
        words = sample.split()
        unique_words.update(words)
        total_words += len(words)
    
    print(f"\nDiversity Metrics:")
    print(f"Unique words ratio: {len(unique_words)/total_words:.2f}")
    print(f"Average length: {total_words/len(samples):.1f} words")

def test_model_coherence(model, preprocessor, device):
    """Test model's ability to maintain emotional and thematic coherence"""
    print("\nCoherence Testing:")
    
    print("\nTesting emotional context")
    prompts = [
        "Love is",
        "Heartbreak feels",
        "Dancing makes me",
        "The rain brings"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        
        # Convert prompt to sequence
        words = prompt.split()
        sequence = [preprocessor.word2idx.get(word, preprocessor.word2idx['<UNK>']) 
                   for word in words]
        sequence = [preprocessor.word2idx['<START>']] + sequence
        sequence = torch.tensor(sequence, device=device)
        
        model.eval()
        with torch.no_grad():
            generated = []
            current_seq = sequence
            hidden = None
            
            for _ in range(30):  # Generate 30 words for coherence testing
                # Forward pass
                output, hidden = model(current_seq, hidden)
                
                # Get next token probabilities
                if len(output.shape) == 3:
                    output = output[:, -1, :]
                output = output / 0.7  # Lower temperature for more coherent output
                probs = torch.softmax(output, dim=-1)
                
                # Sample next token
                if len(probs.shape) > 1:
                    probs = probs[0]
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == preprocessor.word2idx['<END>']:
                    break
                
                # Use get() with <UNK> as default for unknown indices
                next_word = preprocessor.idx2word.get(str(next_token.item()), '<UNK>')
                if next_word == '<UNK>':
                    continue
                
                generated.append(next_word)
                current_seq = next_token
            
            print(' '.join(generated))

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    lyrics = load_and_preprocess_data()
    if not lyrics:
        print("Failed to load data. Exiting...")
        return
    
    # Initialize preprocessor and load saved vocabulary
    print("Initializing preprocessor...")
    preprocessor = TextPreprocessor(min_freq=2)
    if os.path.exists('vocab.json'):
        print("Loading saved vocabulary...")
        with open('vocab.json', 'r') as f:
            vocab_data = json.load(f)
            preprocessor.word2idx = vocab_data['word2idx']
            preprocessor.idx2word = vocab_data['idx2word']
            preprocessor.vocab_size = vocab_data['vocab_size']
        print(f"Loaded vocabulary size: {preprocessor.vocab_size}")
    else:
        print("Building new vocabulary...")
        preprocessor.build_vocab(lyrics)
        print(f"New vocabulary size: {preprocessor.vocab_size}")
    
    # Initialize model
    print("Initializing model...")
    model = LyricsGenerator(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=64,  # Match training parameters
        hidden_dim=128,    # Match training parameters
        n_layers=1,        # Match training parameters
        dropout=0.3        # Match training parameters
    ).to(device)
    
    # Load trained weights
    if os.path.exists('best_model.pth'):
        print("Loading trained model...")
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    else:
        print("No trained model found. Please train the model first.")
        return
    
    # Run tests
    print("\nRunning model tests...")
    
    # Basic generation test
    test_model_basic(model, preprocessor, device)
    
    # Diversity test
    test_model_diversity(model, preprocessor, device)
    
    # Coherence test
    test_model_coherence(model, preprocessor, device)

if __name__ == '__main__':
    main()
