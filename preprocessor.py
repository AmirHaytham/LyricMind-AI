import json
import torch
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with special tokens."""
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<EOS>': 3
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.word_to_index[token] = idx
            self.index_to_word[idx] = token
            
        self.vocab_size = len(self.special_tokens)
        
    def load_vocabulary(self, vocab_path):
        """Load vocabulary from a JSON file."""
        try:
            logger.info(f"Loading vocabulary from {vocab_path}")
            
            # Default vocabulary if file doesn't exist
            default_vocab = {
                "words": ["i", "love", "you", "the", "music", "dance", "heart", "soul", 
                         "dream", "life", "night", "day", "sun", "moon", "star", "sky",
                         "rain", "wind", "fire", "earth", "sea", "mountain", "river", 
                         "forest", "flower", "bird", "song", "melody", "rhythm", "beat",
                         "voice", "silence", "time", "space", "light", "dark", "shadow",
                         "memory", "hope", "faith", "truth", "lie", "joy", "pain", 
                         "laugh", "cry", "smile", "tear", "kiss", "touch", "feel",
                         "think", "know", "believe", "trust", "doubt", "fear", "brave",
                         "strong", "weak", "live", "die", "begin", "end", "forever",
                         "never", "always", "sometimes", "maybe", "yes", "no", "hello",
                         "goodbye", "sorry", "thank", "please", "want", "need", "give",
                         "take", "make", "break", "build", "destroy", "create", "imagine",
                         "remember", "forget", "miss", "find", "lose", "seek", "hide",
                         "run", "walk", "dance", "sing", "play", "stop", "start", "wait",
                         "move", "stay", "come", "go", "return", "leave", "follow", "lead"],
                "special_tokens": ["<PAD>", "<UNK>", "<START>", "<EOS>"]
            }
            
            if not os.path.exists(vocab_path):
                logger.warning(f"Vocabulary file not found at {vocab_path}, using default vocabulary")
                vocab_data = default_vocab
            else:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                
            # Reset vocabulary but keep special tokens
            self.word_to_index = {k: v for k, v in self.special_tokens.items()}
            self.index_to_word = {v: k for k, v in self.special_tokens.items()}
            
            # Add words from vocabulary
            current_idx = len(self.special_tokens)
            for word in vocab_data['words']:
                word = word.lower()  # Convert to lowercase
                if word not in self.word_to_index:
                    self.word_to_index[word] = current_idx
                    self.index_to_word[current_idx] = word
                    current_idx += 1
                    
            self.vocab_size = current_idx
            logger.info(f"Vocabulary loaded. Size: {self.vocab_size}")
            
        except Exception as e:
            logger.error(f"Error loading vocabulary: {str(e)}")
            raise
            
    def text_to_sequence(self, text):
        """Convert text to sequence of indices."""
        # Tokenize and convert to lowercase
        words = text.lower().split()
        
        # Create sequence with start token
        sequence = [self.special_tokens['<START>']]
        
        # Add word indices
        for word in words:
            if word in self.word_to_index:
                sequence.append(self.word_to_index[word])
            else:
                sequence.append(self.special_tokens['<UNK>'])
                logger.warning(f"Unknown word encountered: {word}")
                
        return sequence
        
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text."""
        words = []
        for idx in sequence:
            if idx in self.index_to_word:
                word = self.index_to_word[idx]
                # Skip special tokens in output
                if word not in self.special_tokens:
                    words.append(word)
            else:
                logger.warning(f"Unknown index encountered: {idx}")
                
        return ' '.join(words)
        
    def save_vocabulary(self, vocab_path):
        """Save vocabulary to a JSON file."""
        try:
            # Get all words except special tokens
            words = [word for word, idx in self.word_to_index.items() 
                    if word not in self.special_tokens]
                    
            vocab_data = {
                'words': words,
                'special_tokens': list(self.special_tokens.keys())
            }
            
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, indent=2)
                
            logger.info(f"Vocabulary saved to {vocab_path}")
            
        except Exception as e:
            logger.error(f"Error saving vocabulary: {str(e)}")
            raise
            
    def get_vocab_size(self):
        """Get the size of the vocabulary."""
        return self.vocab_size
