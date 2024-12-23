"""Test configuration settings."""

# API Test Settings
API_TEST_CONFIG = {
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max request size
    'TESTING': True,
    'DEBUG': True
}

# Test Data
TEST_PROMPTS = [
    'I love you',
    'In the midnight hour',
    'Dancing in the rain',
    'Under the stars tonight',
    'Walking down memory lane'
]

TEST_GENRES = ['pop', 'rock', 'hip-hop', 'country', 'jazz']

# Performance Test Settings
PERFORMANCE_CONFIG = {
    'MAX_RESPONSE_TIME': 5.0,  # seconds
    'MAX_MEMORY_INCREASE': 100 * 1024 * 1024,  # 100MB
    'CONCURRENT_REQUESTS': 5,
    'LOAD_TEST_DURATION': 60  # seconds
}

# Error Messages
ERROR_MESSAGES = {
    'NO_PROMPT': 'Please enter some starting lyrics',
    'INVALID_GENRE': 'Invalid genre selected',
    'MODEL_ERROR': 'Error generating lyrics',
    'INVALID_LENGTH': 'Invalid max_length parameter',
    'INVALID_TEMPERATURE': 'Invalid temperature parameter'
}
