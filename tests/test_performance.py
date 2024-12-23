import unittest
import sys
import os
import time
import threading
import queue
import psutil
import json
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from tests.config import PERFORMANCE_CONFIG, TEST_PROMPTS

class TestLyricMindPerformance(unittest.TestCase):
    def setUp(self):
        """Set up test client and other test variables."""
        self.app = app.test_client()
        self.app.testing = True
        self.process = psutil.Process(os.getpid())
        
    def test_response_time_distribution(self):
        """Test response time distribution across multiple requests."""
        prompts = [
            'I love you',
            'In the midnight hour',
            'Dancing in the rain',
            'Under the stars tonight',
            'Walking down memory lane'
        ]
        
        response_times = []
        for prompt in prompts:
            start_time = time.time()
            response = self.app.post('/generate', json={'prompt': prompt})
            end_time = time.time()
            
            self.assertEqual(response.status_code, 503)  # Service Unavailable when model not initialized
            response_times.append(end_time - start_time)
            
        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        # Log results for analysis
        print("\nResponse Time Distribution Test Results:")
        print(f"Average Response Time: {avg_time:.2f}s")
        print(f"Maximum Response Time: {max_time:.2f}s")
        
        # Assertions
        self.assertLess(avg_time, 1.0)  # Average response time should be under 1 second
        self.assertLess(max_time, 2.0)  # Maximum response time should be under 2 seconds
        
    def test_memory_leak(self):
        """Test for memory leaks during repeated requests."""
        initial_memory = self.process.memory_info().rss
        memory_usage = [initial_memory]
        
        for _ in range(10):  # Make 10 requests
            test_data = {
                'prompt': TEST_PROMPTS[0],
                'genre': 'pop',
                'max_length': 50,
                'temperature': 0.5
            }
            
            response = self.app.post('/generate',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
            self.assertEqual(response.status_code, 200)
            
            current_memory = self.process.memory_info().rss
            memory_usage.append(current_memory)
            
        # Calculate memory growth
        memory_growth = memory_usage[-1] - memory_usage[0]
        avg_growth_per_request = memory_growth / 10
        
        print(f"\nMemory Usage Statistics:")
        print(f"Initial Memory: {initial_memory / 1024 / 1024:.2f}MB")
        print(f"Final Memory: {memory_usage[-1] / 1024 / 1024:.2f}MB")
        print(f"Average Growth per Request: {avg_growth_per_request / 1024 / 1024:.2f}MB")
        
        self.assertLess(memory_growth, PERFORMANCE_CONFIG['MAX_MEMORY_INCREASE'])
        
    def test_concurrent_load(self):
        """Test system under concurrent load."""
        results = queue.Queue()
        
        def make_request():
            try:
                test_data = {
                    'prompt': TEST_PROMPTS[0],
                    'genre': 'pop',
                    'max_length': 50,
                    'temperature': 0.5
                }
                
                start_time = time.time()
                response = self.app.post('/generate',
                                       data=json.dumps(test_data),
                                       content_type='application/json')
                end_time = time.time()
                
                results.put({
                    'status_code': response.status_code,
                    'response_time': end_time - start_time
                })
            except Exception as e:
                results.put({
                    'error': str(e)
                })
                
        # Create thread pool
        with ThreadPoolExecutor(max_workers=PERFORMANCE_CONFIG['CONCURRENT_REQUESTS']) as executor:
            # Submit concurrent requests
            futures = [executor.submit(make_request) 
                      for _ in range(PERFORMANCE_CONFIG['CONCURRENT_REQUESTS'])]
            
        # Analyze results
        success_count = 0
        total_response_time = 0
        
        while not results.empty():
            result = results.get()
            if 'error' in result:
                print(f"Request error: {result['error']}")
                continue
                
            if result['status_code'] == 200:
                success_count += 1
                total_response_time += result['response_time']
                
        success_rate = (success_count / PERFORMANCE_CONFIG['CONCURRENT_REQUESTS']) * 100
        avg_response_time = total_response_time / success_count if success_count > 0 else float('inf')
        
        print(f"\nConcurrent Load Test Results:")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Average Response Time: {avg_response_time:.2f}s")
        
        self.assertGreaterEqual(success_rate, 90)  # At least 90% success rate
        self.assertLess(avg_response_time, PERFORMANCE_CONFIG['MAX_RESPONSE_TIME'])
        
    def test_load_over_time(self):
        """Test system performance over an extended period."""
        start_time = time.time()
        request_count = 0
        errors = 0
        
        while time.time() - start_time < PERFORMANCE_CONFIG['LOAD_TEST_DURATION']:
            test_data = {
                'prompt': TEST_PROMPTS[request_count % len(TEST_PROMPTS)],
                'genre': 'pop',
                'max_length': 50,
                'temperature': 0.5
            }
            
            try:
                response = self.app.post('/generate',
                                       data=json.dumps(test_data),
                                       content_type='application/json')
                if response.status_code != 200:
                    errors += 1
            except Exception:
                errors += 1
                
            request_count += 1
            
        duration = time.time() - start_time
        requests_per_second = request_count / duration
        error_rate = (errors / request_count) * 100 if request_count > 0 else 100
        
        print(f"\nLoad Over Time Test Results:")
        print(f"Duration: {duration:.2f}s")
        print(f"Total Requests: {request_count}")
        print(f"Requests per Second: {requests_per_second:.2f}")
        print(f"Error Rate: {error_rate:.2f}%")
        
        self.assertLess(error_rate, 10)  # Less than 10% error rate
        self.assertGreater(requests_per_second, 0.5)  # At least 0.5 requests per second

if __name__ == '__main__':
    unittest.main()
