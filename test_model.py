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
            current_seq = torch.tensor([sequence], device=device)
            
            # Generate
            model.eval()
            with torch.no_grad():
                generated = []
                hidden = None
                
                for _ in range(50):  # Generate up to 50 words
                    output, hidden = model(current_seq, hidden)
                    output = output[:, -1, :] / temp
                    probs = torch.softmax(output, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    if next_token.item() == preprocessor.word2idx['<END>']:
                        break
                        
                    generated.append(preprocessor.idx2word[next_token.item()])
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
    
    # Generate multiple samples
    for i in range(n_samples):
        current_seq = torch.tensor([sequence], device=device)
        model.eval()
        with torch.no_grad():
            generated = []
            hidden = None
            
            for _ in range(50):
                output, hidden = model(current_seq, hidden)
                output = output[:, -1, :] / temperature
                probs = torch.softmax(output, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == preprocessor.word2idx['<END>']:
                    break
                    
                generated.append(preprocessor.idx2word[next_token.item()])
                current_seq = next_token.unsqueeze(0)
        
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
    """Test model's ability to maintain coherence"""
    print("\nCoherence Testing:")
    
    test_cases = [
        ("Love is", "Testing emotional context"),
        ("In the night", "Testing setting/atmosphere"),
        ("She was", "Testing character/narrative"),
        ("The rhythm of", "Testing musical context")
    ]
    
    for prompt, desc in test_cases:
        print(f"\n{desc}")
        print(f"Prompt: {prompt}")
        
        # Convert prompt to sequence
        sequence = [preprocessor.word2idx['<START>']]
        sequence += [preprocessor.word2idx.get(word, preprocessor.word2idx['<UNK>']) 
                    for word in prompt.split()]
        current_seq = torch.tensor([sequence], device=device)
        
        # Generate
        model.eval()
        with torch.no_grad():
            generated = []
            hidden = None
            
            for _ in range(50):
                output, hidden = model(current_seq, hidden)
                output = output[:, -1, :] / 0.7  # Use lower temperature for coherence
                probs = torch.softmax(output, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == preprocessor.word2idx['<END>']:
                    break
                    
                generated.append(preprocessor.idx2word[next_token.item()])
                current_seq = next_token.unsqueeze(0)
        
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
    
    # Initialize preprocessor
    print("Initializing preprocessor...")
    preprocessor = TextPreprocessor(min_freq=2)
    preprocessor.build_vocab(lyrics)
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    
    # Initialize model
    print("Initializing model...")
    model = LyricsGenerator(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        n_layers=2,
        dropout=0.5
    ).to(device)
    
    # Load trained weights
    if os.path.exists('model_checkpoint.pth'):
        print("Loading trained model...")
        checkpoint = torch.load('model_checkpoint.pth', map_location=device)
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
