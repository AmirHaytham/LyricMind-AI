# Contributing to LyricMind-AI

First off, thank you for considering contributing to LyricMind-AI! It's people like you that make LyricMind-AI such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check [this list](../../issues) as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots if possible

### Suggesting Enhancements

If you have a suggestion for a new feature or enhancement, first check if it's already been suggested. If it hasn't, feel free to create a new issue with the following information:

* Use a clear and descriptive title
* Provide a detailed description of the suggested enhancement
* Explain why this enhancement would be useful
* List some examples of how it would be used

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Process

1. Clone the repository
```bash
git clone https://github.com/AmirHaytham/LyricMind-AI.git
cd LyricMind-AI
```

2. Create a branch
```bash
git checkout -b feature/my-feature
```

3. Make your changes and commit them
```bash
git add .
git commit -m "Add some feature"
```

4. Push to your fork
```bash
git push origin feature/my-feature
```

5. Open a Pull Request

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Styleguide

* Follow PEP 8
* Use docstrings for functions and classes
* Use type hints where possible
* Keep functions focused and small
* Write meaningful variable names

### Documentation Styleguide

* Use Markdown
* Reference functions and classes with backticks
* Include code examples where relevant
* Keep explanations clear and concise

## Project Structure

```
LyricMind-AI/
├── app.py                 # Flask web application
├── model.py              # Neural network architecture
├── train.py             # Model training script
├── data_preprocessing.py # Data preparation utilities
├── tests/              # Test suites
└── docs/               # Documentation
```

## Testing

Run the test suite:
```bash
pytest
```

Run specific tests:
```bash
pytest tests/test_model.py
```

## Additional Notes

### Issue and Pull Request Labels

| Label | Description |
|-------|-------------|
| bug | Something isn't working |
| enhancement | New feature or request |
| good first issue | Good for newcomers |
| help wanted | Extra attention is needed |
| documentation | Improvements to documentation |

## Recognition

Contributors will be featured in our README.md and on our website.

## Questions?

Feel free to contact the core team at team@lyricmind.ai

Thank you for contributing! 🎵
