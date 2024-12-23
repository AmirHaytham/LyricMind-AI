import unittest
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
import torch

class TestLyricMindAPI(unittest.TestCase):
    def setUp(self):
        """Set up test client and other test variables."""
        self.app = app.test_client()
        self.app.testing = True
        
    def test_home_endpoint(self):
        """Test the home endpoint returns 200."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        
    def test_generate_endpoint_valid_input(self):
        """Test lyrics generation with valid input."""
        test_data = {
            'prompt': 'I love you',
            'genre': 'pop',
            'max_length': 50,
            'temperature': 0.5
        }
        
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('lyrics', data)
        self.assertIsInstance(data['lyrics'], str)
        self.assertTrue(len(data['lyrics']) > 0)
        
    def test_generate_endpoint_no_prompt(self):
        """Test lyrics generation with missing prompt."""
        test_data = {
            'genre': 'pop',
            'max_length': 50,
            'temperature': 0.5
        }
        
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
    def test_generate_endpoint_invalid_max_length(self):
        """Test lyrics generation with invalid max_length."""
        test_data = {
            'prompt': 'I love you',
            'genre': 'pop',
            'max_length': 'invalid',
            'temperature': 0.5
        }
        
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIsNone(data.get('lyrics'))
        
    def test_generate_endpoint_invalid_temperature(self):
        """Test lyrics generation with invalid temperature."""
        test_data = {
            'prompt': 'I love you',
            'genre': 'pop',
            'max_length': 50,
            'temperature': 'invalid'
        }
        
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIsNone(data.get('lyrics'))
        
    def test_generate_endpoint_long_prompt(self):
        """Test lyrics generation with a very long prompt."""
        test_data = {
            'prompt': 'I love you ' * 100,  # Very long prompt
            'genre': 'pop',
            'max_length': 50,
            'temperature': 0.5
        }
        
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('lyrics', data)
        
    def test_generate_endpoint_special_characters(self):
        """Test lyrics generation with special characters in prompt."""
        test_data = {
            'prompt': 'I love you!@#$%^&*()',
            'genre': 'pop',
            'max_length': 50,
            'temperature': 0.5
        }
        
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('lyrics', data)
        
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import threading
        import queue
        
        q = queue.Queue()
        def make_request():
            test_data = {
                'prompt': 'I love you',
                'genre': 'pop',
                'max_length': 50,
                'temperature': 0.5
            }
            response = self.app.post('/generate',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
            q.put(response.status_code)
            
        # Create 5 concurrent requests
        threads = []
        for _ in range(5):
            t = threading.Thread(target=make_request)
            t.start()
            threads.append(t)
            
        # Wait for all threads to complete
        for t in threads:
            t.join()
            
        # Check all responses
        while not q.empty():
            self.assertEqual(q.get(), 200)
            
    def test_memory_usage(self):
        """Test memory usage during generation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple requests
        for _ in range(5):
            test_data = {
                'prompt': 'I love you',
                'genre': 'pop',
                'max_length': 50,
                'temperature': 0.5
            }
            response = self.app.post('/generate',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
            self.assertEqual(response.status_code, 200)
            
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be less than 100MB
        self.assertLess(memory_increase, 100 * 1024 * 1024)
        
    def test_response_time(self):
        """Test response time for generation."""
        import time
        
        test_data = {
            'prompt': 'I love you',
            'genre': 'pop',
            'max_length': 50,
            'temperature': 0.5
        }
        
        start_time = time.time()
        response = self.app.post('/generate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        end_time = time.time()
        
        response_time = end_time - start_time
        # Response should be under 5 seconds
        self.assertLess(response_time, 5.0)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
