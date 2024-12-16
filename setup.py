from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lyricmind-ai",
    version="0.1.0",
    author="Amir Haytham",
    author_email="amir.haytham.salama@gmail.com",
    description="Intelligent lyrics generation using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmirHaytham/LyricMind-AI",
    project_urls={
        "Bug Tracker": "https://github.com/AmirHaytham/LyricMind-AI/issues",
        "Documentation": "https://github.com/AmirHaytham/LyricMind-AI/tree/main/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "lyricmind"},
    packages=find_packages(where="lyricmind"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.1",
        "flask>=2.0.1",
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "nltk>=3.6.3",
        "tqdm>=4.62.3",
        "requests>=2.26.0",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.7b0",
            "flake8>=3.9.2",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "lyricmind=lyricmind.cli:main",
        ],
    },
)
