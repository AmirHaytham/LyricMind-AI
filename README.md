# 🎵 LyricMind-AI: Creative Song Lyrics Generator

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/AMIRHaytham/lyricMind-AI)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Twitter](https://img.shields.io/badge/twitter-@AmirHaytham__-1DA1F2)](https://twitter.com/AmirHaytham_)

[Project Banner Image]

LyricMind-AI is an advanced deep learning model that generates creative and contextually relevant song lyrics. Built with PyTorch and Flask, it uses LSTM architecture to understand and generate human-like lyrics across different musical genres.

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Model Architecture](#-model-architecture)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Features

- **Creative Lyrics Generation**: Generate unique and contextually relevant lyrics from any prompt
- **Genre-Aware**: Supports multiple music genres including Pop, Rock, Hip Hop, Country, and Jazz
- **Temperature Control**: Adjust creativity vs. coherence with temperature parameter
- **User-Friendly Interface**: Clean, modern web interface for easy interaction
- **Real-Time Generation**: Fast response times with asynchronous processing

[Screenshot of Web Interface]

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git
- 4GB+ RAM recommended
- CUDA-capable GPU (optional, for faster training)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AMIRHaytham/lyricMind-AI.git
   cd lyricMind-AI
   ```

2. **Set Up Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Data**
   - Download the datasets:
     - [Spotify Million Song Dataset](https://www.kaggle.com/spotify/million-song-dataset)
     - [Billboard Top 500](https://www.billboard.com/charts/hot-100)
   - Place the files in the project root directory

## 💻 Usage

### Web Interface
1. Start the server:
   ```bash
   python app.py
   ```
2. Open `http://localhost:5000` in your browser
3. Enter your prompt and adjust parameters
4. Click "Generate Lyrics"

### Python API
```python
from lyricmind import LyricGenerator

# Initialize generator
generator = LyricGenerator(
    genre='pop',
    model_path='best_model.pth',
    vocab_path='vocab.json'
)

# Generate lyrics
lyrics = generator.generate(
    prompt="In the midnight hour",
    temperature=0.7,
    max_length=100
)
```

## 🔌 API Reference

### RESTful API Endpoints

#### POST /generate
Generate lyrics from a prompt.

**Request Body:**
```json
{
    "prompt": "In the midnight hour",
    "temperature": 0.7,
    "max_length": 100,
    "genre": "pop"
}
```

**Response:**
```json
{
    "lyrics": "Generated lyrics...",
    "error": null
}
```

For detailed API documentation, see our [API Reference Guide](docs/api.md).

## 🧠 Model Architecture

### Overview
LyricMind-AI uses a deep learning architecture based on LSTM networks:

```
Input Text → Embedding → LSTM → Dropout → Linear → Softmax → Output
```

### Components
- **Embedding Layer**: 64 dimensions, ~50,000 word vocabulary
- **LSTM Layer**: 128 hidden units, single layer, 0.3 dropout
- **Output Layer**: Linear transformation with softmax activation

### Training Details
- Dataset: 1M+ lyrics across multiple genres
- Training Parameters:
  - Batch Size: 64
  - Learning Rate: 0.001
  - Epochs: 50
  - Optimizer: Adam

## 👥 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code of Conduct
- Development Process
- Pull Request Process
- Style Guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Created by [Amir Haytham](https://twitter.com/AmirHaytham_) | [GitHub](https://github.com/AMIRHaytham)
