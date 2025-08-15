"""
Setup script for Smart Document Analyzer & Knowledge Portal
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="smart-document-analyzer",
    version="1.0.0",
    author="Smart Document Analyzer Team",
    author_email="contact@smartdocanalyzer.com",
    description="A comprehensive ML/NLP backend system for intelligent document processing, analysis, and retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smart-document-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "python-multipart>=0.0.6",
        ],
        "gpu": [
            "torch[cuda]",
            "faiss-gpu>=1.7.4",
        ]
    },
    entry_points={
        "console_scripts": [
            "sda-train-classifier=scripts.train_classifier:main",
            "sda-demo=scripts.inference_examples:main",
            "sda-api=scripts.api_server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
        "config": ["*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords="nlp, machine learning, document processing, text analysis, semantic search, summarization, classification, named entity recognition",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/smart-document-analyzer/issues",
        "Source": "https://github.com/yourusername/smart-document-analyzer",
        "Documentation": "https://smart-document-analyzer.readthedocs.io/",
    },
)
