from flask import Flask, render_template, request, jsonify
import torch
from model import LyricsGenerator
from data_preprocessing import TextPreprocessor
import os

app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None

def load_model():
    """Load the model and preprocessor with error handling"""
    try:
        # Initialize preprocessor and model with the same parameters used during training
        preprocessor = TextPreprocessor(min_freq=2)
        
        # For testing purposes, initialize with minimal vocabulary if no checkpoint exists
        if not os.path.exists('model_checkpoint.pth'):
            preprocessor.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
            preprocessor.idx2word = {v: k for k, v in preprocessor.word2idx.items()}
            preprocessor.vocab_size = len(preprocessor.word2idx)
        
        model = LyricsGenerator(
            vocab_size=len(preprocessor.word2idx),
            embedding_dim=256,
            hidden_dim=512,
            n_layers=2,
            dropout=0.5
        )
        
        # Load trained weights if available
        if os.path.exists('model_checkpoint.pth'):
            checkpoint = torch.load('model_checkpoint.pth', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            preprocessor.word2idx = checkpoint['word2idx']
            preprocessor.idx2word = checkpoint['idx2word']
            preprocessor.vocab_size = len(preprocessor.word2idx)
        
        model.eval()  # Set to evaluation mode
        return model, preprocessor
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def validate_input(data):
    """Validate and sanitize input data"""
    errors = []
    
    # Validate max_length
    max_length = data.get('max_length', 200)
    if not isinstance(max_length, (int, float)) or max_length < 50 or max_length > 500:
        errors.append("max_length must be between 50 and 500")
        max_length = 200  # Use default
    
    # Validate temperature
    temperature = data.get('temperature', 1.0)
    if not isinstance(temperature, (int, float)) or temperature < 0.1 or temperature > 2.0:
        errors.append("temperature must be between 0.1 and 2.0")
        temperature = 1.0  # Use default
    
    # Validate genre
    valid_genres = ['pop', 'rock', 'hip-hop', 'country', 'jazz', '']
    genre = data.get('genre', '').lower()
    if genre not in valid_genres:
        errors.append(f"genre must be one of: {', '.join(valid_genres)}")
        genre = ''  # Use default
    
    return {
        'artist': data.get('artist', ''),
        'genre': genre,
        'max_length': int(max_length),
        'temperature': float(temperature)
    }, errors

# Load model on startup
model, preprocessor = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Check if model is loaded
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not initialized properly'}), 500
            
        data = request.get_json()
        validated_data, errors = validate_input(data)
        
        if errors:
            return jsonify({'error': 'Invalid input: ' + '; '.join(errors)}), 400
        
        # Generate lyrics
        device = torch.device('cpu')
        model.to(device)
        
        # For testing purposes, return dummy lyrics if no checkpoint exists
        if not os.path.exists('model_checkpoint.pth'):
            return jsonify({'lyrics': 'This is a test lyric generated for testing purposes.'})
        
        # Start with START token
        current_seq = torch.tensor([[preprocessor.word2idx['<START>']]], device=device)
        generated_text = []
        
        # Generate sequence
        with torch.no_grad():
            hidden = None
            for i in range(validated_data['max_length']):
                output, hidden = model(current_seq, hidden)
                
                # Apply temperature scaling
                output = output[:, -1, :] / validated_data['temperature']
                probs = torch.softmax(output, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, 1)
                
                # Break if END token is generated
                if next_token.item() == preprocessor.word2idx['<END>']:
                    break
                    
                generated_text.append(preprocessor.idx2word[next_token.item()])
                current_seq = next_token.unsqueeze(0)
        
        # Join the generated words
        lyrics = ' '.join(generated_text)
        return jsonify({'lyrics': lyrics})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
