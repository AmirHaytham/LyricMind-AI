.PHONY: install test lint clean run

# Variables
PYTHON = python
PIP = pip
FLASK = flask
TEST = pytest
LINT = flake8

# Installation
install:
	$(PIP) install -r requirements.txt

# Testing
test:
	$(TEST) tests/

# Linting
lint:
	$(LINT) .

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".eggs" -exec rm -r {} +

# Run development server
run:
	$(PYTHON) app.py

# Train model
train:
	$(PYTHON) train.py

# Generate test coverage report
coverage:
	coverage run -m pytest tests/
	coverage report
	coverage html

# Create distribution package
dist:
	$(PYTHON) setup.py sdist bdist_wheel
