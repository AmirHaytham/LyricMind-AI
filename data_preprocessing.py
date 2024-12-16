import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import torch
from torch.utils.data import Dataset, DataLoader

class TextPreprocessor:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?\'"-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        # Count word frequencies
        word_freq = Counter()
        for text in texts:
            words = self.clean_text(text).split()
            word_freq.update(words)
        
        # Create vocabulary (only include words that appear more than min_freq times)
        vocab_words = ['<PAD>', '<UNK>', '<START>', '<END>'] + \
                     [word for word, freq in word_freq.items() if freq >= self.min_freq]
        
        # Create word to index mapping
        self.word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = self.clean_text(text).split()
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]

class LyricsDataset(Dataset):
    def __init__(self, texts, preprocessor, seq_length=50):
        self.preprocessor = preprocessor
        self.seq_length = seq_length
        
        # Process all texts
        self.sequences = []
        for text in texts:
            sequence = self.preprocessor.text_to_sequence(text)
            # Add start and end tokens
            sequence = [self.preprocessor.word2idx['<START>']] + sequence + [self.preprocessor.word2idx['<END>']]
            self.sequences.append(sequence)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Pad or truncate sequence to fixed length
        if len(sequence) > self.seq_length:
            sequence = sequence[:self.seq_length]
        else:
            sequence = sequence + [self.preprocessor.word2idx['<PAD>']] * (self.seq_length - len(sequence))
        
        # Create input and target sequences for training
        x = torch.tensor(sequence[:-1])
        y = torch.tensor(sequence[1:])
        
        return x, y

def prepare_data(spotify_path, top500_path, test_size=0.2, min_freq=2, seq_length=50, batch_size=32):
    """
    Prepare data for training the lyrics generation model
    
    Args:
        spotify_path (str): Path to Spotify dataset
        top500_path (str): Path to Top 500 songs dataset
        test_size (float): Proportion of data to use for testing
        min_freq (int): Minimum frequency for words to be included in vocabulary
        seq_length (int): Length of sequences for training
        batch_size (int): Batch size for training
        
    Returns:
        dict: Dictionary containing DataLoaders and preprocessor
    """
    # Load datasets
    spotify_df = pd.read_csv(spotify_path, encoding='latin1')
    top500_df = pd.read_csv(top500_path, encoding='latin1')
    
    # Combine lyrics from both datasets
    all_lyrics = pd.concat([spotify_df['text'], top500_df['description']]).dropna()
    
    # Create preprocessor and build vocabulary
    preprocessor = TextPreprocessor(min_freq=min_freq)
    preprocessor.build_vocab(all_lyrics)
    
    # Split data into train and test sets
    train_texts, test_texts = train_test_split(all_lyrics, test_size=test_size, random_state=42)
    
    # Create datasets
    train_dataset = LyricsDataset(train_texts, preprocessor, seq_length=seq_length)
    test_dataset = LyricsDataset(test_texts, preprocessor, seq_length=seq_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'preprocessor': preprocessor
    }

if __name__ == '__main__':
    # Example usage
    data = prepare_data(
        spotify_path='Spotify Million Song Dataset_exported.csv',
        top500_path='Top 500 Songs.csv'
    )
    print(f"Vocabulary size: {data['preprocessor'].vocab_size}")
    print(f"Number of training batches: {len(data['train_loader'])}")
    print(f"Number of testing batches: {len(data['test_loader'])}")
