import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import LyricsGenerator
from data_preprocessing import prepare_data
import json
import os

class TrainingConfig:
    def __init__(self):
        self.embedding_dim = 64
        self.hidden_dim = 128
        self.n_layers = 1
        self.dropout = 0.3
        self.learning_rate = 0.001
        self.epochs = 5
        self.batch_size = 8
        self.seq_length = 30
        self.min_freq = 3
        self.test_size = 0.2
        
def train_model(config):
    # Prepare data
    data = prepare_data(
        spotify_path='Spotify Million Song Dataset_exported.csv',
        top500_path='Top 500 Songs.csv',
        test_size=config.test_size,
        min_freq=config.min_freq,
        seq_length=config.seq_length,
        batch_size=config.batch_size
    )
    
    # Initialize model
    model = LyricsGenerator(
        vocab_size=data['preprocessor'].vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout
    )
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=data['preprocessor'].word2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = tqdm(data['train_loader'], desc=f'Epoch {epoch+1}/{config.epochs}')
        
        for batch_x, batch_y in train_batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(batch_x)
            
            loss = criterion(output.view(-1, data['preprocessor'].vocab_size), batch_y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches.set_postfix({'train_loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(data['train_loader'])
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_test_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data['test_loader']:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                output, _ = model(batch_x)
                loss = criterion(output.view(-1, data['preprocessor'].vocab_size), batch_y.view(-1))
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(data['test_loader'])
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1}/{config.epochs}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average test loss: {avg_test_loss:.4f}')
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }, 'best_model.pth')
            
            # Save preprocessor vocabulary
            vocab_data = {
                'word2idx': data['preprocessor'].word2idx,
                'idx2word': data['preprocessor'].idx2word,
                'vocab_size': data['preprocessor'].vocab_size
            }
            with open('vocab.json', 'w') as f:
                json.dump(vocab_data, f)
    
    return model, data['preprocessor'], train_losses, test_losses

def generate_lyrics(model, preprocessor, artist_style='', genre='', max_length=200, temperature=1.0):
    """Generate lyrics using the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Start with START token
    current_seq = [preprocessor.word2idx['<START>']]
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            x = torch.tensor([current_seq]).to(device)
            
            # Get model predictions
            output, _ = model(x)
            
            # Apply temperature
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            
            # Sample from the distribution
            next_word_idx = torch.multinomial(probs, 1).item()
            
            # Stop if we generate END token
            if next_word_idx == preprocessor.word2idx['<END>']:
                break
                
            current_seq.append(next_word_idx)
    
    # Convert indices back to words
    generated_words = [preprocessor.idx2word[idx] for idx in current_seq 
                      if idx not in [preprocessor.word2idx['<START>'], preprocessor.word2idx['<END>']]]
    
    return ' '.join(generated_words)

if __name__ == '__main__':
    # Train model
    config = TrainingConfig()
    model, preprocessor, train_losses, test_losses = train_model(config)
    
    # Generate sample lyrics
    sample_lyrics = generate_lyrics(model, preprocessor)
    print("\nGenerated Sample Lyrics:")
    print(sample_lyrics)
