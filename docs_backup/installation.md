# Installation Guide

This guide will help you set up LyricMind-AI on your local machine.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- 4GB+ RAM recommended
- CUDA-capable GPU (optional, for faster training)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirHaytham/LyricMind-AI.git
cd LyricMind-AI
```

### 2. Set Up Virtual Environment (Recommended)

#### On Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### On macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Datasets
1. Download the required datasets:
   - [Spotify Million Song Dataset](https://www.kaggle.com/spotify/million-song-dataset)
   - [Billboard Top 500](https://www.billboard.com/charts/hot-100)

2. Place the dataset files in the project root:
   ```
   LyricMind-AI/
   ├── Spotify Million Song Dataset_exported.csv
   └── Top 500 Songs.csv
   ```

### 5. Train the Model (Optional)

If you want to train the model from scratch:
```bash
python train.py
```

### 6. Start the Web Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## GPU Support (Optional)

For GPU acceleration:

1. Install CUDA Toolkit 11.x from [NVIDIA's website](https://developer.nvidia.com/cuda-toolkit)
2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Common Issues

### Issue: Out of Memory
**Solution**: Reduce batch size in `train.py`:
```python
self.batch_size = 8  # Reduce this value
```

### Issue: CUDA not found
**Solution**: Verify CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

- Check out the [Model Architecture](model.md)
- Read the [API Documentation](api.md)
- See [Contributing Guidelines](../CONTRIBUTING.md)
