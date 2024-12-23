import torch
import torch.nn as nn

class LyricsGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # Ensure input is 2D [batch_size, seq_len]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Apply embedding and dropout
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # LSTM layer
        output, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout to output
        output = self.dropout(output)
        
        # Final linear layer
        prediction = self.fc(output)
        
        # Apply log softmax for better numerical stability
        prediction = torch.log_softmax(prediction, dim=-1)
        
        return prediction, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        n_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size
        
        hidden = (torch.zeros(n_layers, batch_size, hidden_dim).to(device),
                 torch.zeros(n_layers, batch_size, hidden_dim).to(device))
        return hidden
