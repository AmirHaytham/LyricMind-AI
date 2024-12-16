import unittest
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

class TestLyricsGenerator(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        """Test if home page loads correctly"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI Lyrics Generator', response.data)

    def test_generate_lyrics_valid_input(self):
        """Test lyrics generation with valid input"""
        test_data = {
            'artist': 'Test Artist',
            'genre': 'pop',
            'max_length': 100,
            'temperature': 1.0
        }
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('lyrics', data)

    def test_generate_lyrics_invalid_input(self):
        """Test lyrics generation with invalid input"""
        test_data = {
            'artist': 'Test Artist',
            'genre': 'invalid_genre',
            'max_length': -1,  # Invalid length
            'temperature': 5.0  # Invalid temperature
        }
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_generate_lyrics_missing_input(self):
        """Test lyrics generation with missing input"""
        test_data = {}  # Empty data
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)  # Should still work with defaults

    def test_model_output_format(self):
        """Test if model output is properly formatted"""
        test_data = {
            'artist': 'Test Artist',
            'genre': 'pop',
            'max_length': 50,
            'temperature': 0.5
        }
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        data = json.loads(response.data)
        self.assertIsInstance(data['lyrics'], str)
        self.assertTrue(len(data['lyrics']) > 0)

if __name__ == '__main__':
    unittest.main()
