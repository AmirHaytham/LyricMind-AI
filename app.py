import os
import json
import torch
import logging
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import LyricsGenerator
from preprocessor import TextPreprocessor

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = app.logger

# Global variables
model = None
preprocessor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model and preprocessor
def init_model():
    """Initialize the model and preprocessor."""
    global model, preprocessor, device
    
    try:
        # Initialize preprocessor
        logger.info("Starting model loading process...")
        preprocessor = TextPreprocessor()
        logger.info("TextPreprocessor initialized")
        
        # Load vocabulary
        vocab_path = os.path.join(os.path.dirname(__file__), 'vocab.json')
        logger.info(f"Loading vocabulary from {vocab_path}")
        preprocessor.load_vocabulary(vocab_path)
        
        # Initialize model
        logger.info("Initializing model...")
        model = LyricsGenerator(
            vocab_size=preprocessor.get_vocab_size(),
            embedding_dim=64,
            hidden_dim=128,
            num_layers=1,
            dropout=0.3
        ).to(device)
        logger.info(f"Model initialized on device: {device}")
        
        # Load model weights if available
        model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
        if os.path.exists(model_path):
            logger.info(f"Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded successfully")
        else:
            logger.warning(f"No model weights found at {model_path}, using initialized weights")
        
        model.eval()
        logger.info("Model set to evaluation mode")
        
        return model, preprocessor
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        logger.error("Traceback: %s", traceback.format_exc())
        return None, None

def generate_lyrics(prompt, max_length=50, temperature=0.7):
    """Generate lyrics given a prompt."""
    try:
        if model is None or preprocessor is None:
            logger.error("Model or preprocessor not initialized")
            return None
            
        # Preprocess prompt
        logger.info(f'Words from prompt: {prompt.split()}')
        input_sequence = preprocessor.text_to_sequence(prompt)
        if not input_sequence:
            logger.error("Failed to convert prompt to sequence")
            return None
            
        logger.info(f'Input sequence: {input_sequence}')
        
        # Convert to tensor
        input_tensor = torch.tensor([input_sequence]).to(device)
        logger.info(f'Input tensor shape: {input_tensor.shape}')
        
        # Generate
        with torch.no_grad():
            generated_sequence = []
            current_input = input_tensor
            
            for _ in range(max_length):
                # Forward pass
                output = model(current_input)
                
                # Get next token probabilities
                next_token_logits = output[0, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=0)
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, 1)
                
                # Check for end of sequence
                if next_token.item() == preprocessor.special_tokens['<EOS>']:
                    break
                    
                # Add to generated sequence
                generated_sequence.append(next_token.item())
                
                # Update input for next iteration
                current_input = torch.cat([current_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
        # Convert sequence back to text
        generated_text = preprocessor.sequence_to_text(generated_sequence)
        if not generated_text:
            logger.warning("Generated empty text")
            return "Could not generate meaningful lyrics. Please try a different prompt."
            
        logger.info(f'Final generated lyrics: {generated_text!r}')
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f'Error in generate_lyrics: {str(e)}')
        logger.error(f'Traceback: {traceback.format_exc()}')
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate lyrics based on input parameters."""
    try:
        data = request.get_json()
        
        # Get parameters with defaults
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 50)
        temperature = data.get('temperature', 0.7)
        genre = data.get('genre', 'pop')  # Default genre
        
        # Validate parameters
        if not prompt:
            return jsonify({
                'error': 'No prompt provided',
                'lyrics': None
            }), 400
            
        try:
            max_length = int(max_length)
            if max_length <= 0 or max_length > 500:
                return jsonify({
                    'error': 'max_length must be between 1 and 500',
                    'lyrics': None
                }), 400
        except ValueError:
            return jsonify({
                'error': 'Invalid max_length value',
                'lyrics': None
            }), 400
            
        try:
            temperature = float(temperature)
            if temperature <= 0 or temperature > 2.0:
                return jsonify({
                    'error': 'temperature must be between 0 and 2.0',
                    'lyrics': None
                }), 400
        except ValueError:
            return jsonify({
                'error': 'Invalid temperature value',
                'lyrics': None
            }), 400
            
        # Log parameters
        logger.info(f"Validated parameters - prompt: '{prompt}', max_length: {max_length}, temperature: {temperature}, genre: {genre}")
        
        # Check if model is initialized
        if model is None or preprocessor is None:
            return jsonify({
                'error': 'Model not initialized. Please try again later.',
                'lyrics': None
            }), 503  # Service Unavailable
        
        # Generate lyrics
        logger.info(f"Generating lyrics with prompt: '{prompt}', max_length: {max_length}, temperature: {temperature}")
        lyrics = generate_lyrics(prompt, max_length, temperature)
        
        if lyrics is None:
            return jsonify({
                'error': 'Failed to generate lyrics',
                'lyrics': None
            }), 500
            
        # Log success
        logger.info(f"Successfully generated lyrics: '{lyrics}'")
        
        return jsonify({
            'error': None,
            'lyrics': lyrics
        }), 200
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'lyrics': None
        }), 500

# Load model on startup
logger.info("\n=== Starting Application ===")
logger.info("Loading model...")
model, preprocessor = init_model()
if model is None or preprocessor is None:
    logger.error("Failed to load model or preprocessor")
else:
    logger.info("Model loaded successfully")

if __name__ == '__main__':
    app.run(debug=True)
