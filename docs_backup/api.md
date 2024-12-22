# API Reference

LyricMind-AI provides both a Python API and a RESTful HTTP API for generating lyrics.

## Python API

### LyricGenerator Class

The main class for generating lyrics.

```python
from lyricmind import LyricGenerator

generator = LyricGenerator(
    genre='pop',
    model_path='path/to/model.pth',
    vocab_path='path/to/vocab.json'
)
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| genre | str | Music genre for generation | 'pop' |
| model_path | str | Path to trained model | 'best_model.pth' |
| vocab_path | str | Path to vocabulary file | 'vocab.json' |
| device | str | 'cuda' or 'cpu' | auto-detected |

#### Methods

##### generate()
Generate lyrics from a prompt.

```python
lyrics = generator.generate(
    prompt="In the midnight hour",
    temperature=0.7,
    max_length=100
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| prompt | str | Starting text | Required |
| temperature | float | Randomness (0.0-1.0) | 0.7 |
| max_length | int | Maximum words | 100 |

##### set_genre()
Change the generation genre.

```python
generator.set_genre('rock')
```

## RESTful API

### Authentication

```bash
curl -X POST https://api.lyricmind.ai/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your_api_key"}'
```

### Generate Lyrics

#### Request

```bash
curl -X POST https://api.lyricmind.ai/generate \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "In the midnight hour",
    "genre": "pop",
    "temperature": 0.7,
    "max_length": 100
  }'
```

#### Response

```json
{
  "status": "success",
  "lyrics": "In the midnight hour, when stars align...",
  "metadata": {
    "genre": "pop",
    "temperature": 0.7,
    "length": 45
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 429 | Rate Limit Exceeded |
| 500 | Server Error |

## WebSocket API

For real-time generation:

```javascript
const ws = new WebSocket('wss://api.lyricmind.ai/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'generate',
    prompt: 'In the midnight hour'
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log(response.lyrics);
};
```

## Rate Limits

- Free tier: 100 requests/day
- Pro tier: 10,000 requests/day
- Enterprise: Custom limits

## Examples

### Python Example

```python
from lyricmind import LyricGenerator

# Initialize generator
generator = LyricGenerator(genre='rock')

# Generate with different temperatures
lyrics_creative = generator.generate(
    prompt="Thunder rolls",
    temperature=0.9
)

lyrics_focused = generator.generate(
    prompt="Thunder rolls",
    temperature=0.3
)
```

### JavaScript Example

```javascript
const LyricMind = require('lyricmind');

const generator = new LyricMind.Generator({
  apiKey: 'your_api_key'
});

generator.generate({
  prompt: "Dancing in the",
  genre: "disco",
  temperature: 0.7
}).then(lyrics => {
  console.log(lyrics);
});
```

## SDK Support

- [Python SDK](https://github.com/AmirHaytham/LyricMind-AI-Python)
- [JavaScript SDK](https://github.com/AmirHaytham/LyricMind-AI-JS)
- [Java SDK](https://github.com/AmirHaytham/LyricMind-AI-Java)
- [Go SDK](https://github.com/AmirHaytham/LyricMind-AI-Go)
