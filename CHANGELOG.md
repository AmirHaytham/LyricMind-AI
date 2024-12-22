# Changelog

All notable changes to LyricMind-AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Update dependencies to fix security vulnerabilities:
  - Update Flask to 3.0.0
  - Update Flask-CORS to 4.0.0
  - Add explicit version pins for all dependencies
  - Add security-related packages (cryptography, urllib3, certifi)
  - Update PyTorch to 2.1.2+cpu
  - Update numpy, pandas, and other data processing libraries

### Added
- Enhanced documentation with comprehensive README
- API endpoint for lyrics generation
- Web interface for easy interaction
- Multi-genre support in model training
- Temperature control for generation creativity

### Changed
- Consolidated documentation into README.md
- Improved error handling in API endpoints
- Enhanced model architecture with better dropout

### Fixed
- Model initialization issues
- API response format consistency
- Input validation in generation endpoint

## [1.0.0] - 2024-12-22

### Added
- Initial release of LyricMind-AI
- LSTM-based lyrics generation model
- Flask web application
- Basic API endpoints
- Dataset preprocessing utilities
- Model training scripts
- Web interface templates
