# Smart Document Analyzer - Makefile
# Common development and deployment tasks

.PHONY: help install install-dev setup clean test lint format run-demo run-api train-sample

# Default target
help:
	@echo "Smart Document Analyzer - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install package and dependencies"
	@echo "  install-dev      Install with development dependencies"
	@echo "  setup            Initial project setup (create directories, download models)"
	@echo ""
	@echo "Development:"
	@echo "  clean           Clean build artifacts and cache files"
	@echo "  test            Run unit tests"
	@echo "  lint            Run code linting"
	@echo "  format          Format code with black and isort"
	@echo ""
	@echo "Demo & Testing:"
	@echo "  run-demo        Run comprehensive demo"
	@echo "  run-api         Start FastAPI server"
	@echo "  train-sample    Train sample classification model"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run Docker container"

# Installation targets
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,api]"
	python -m spacy download en_core_web_sm
	python -m spacy download en_core_web_lg
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

setup:
	mkdir -p models data logs temp chroma_db indices
	@echo "Project directories created"
	@echo "Run 'make install' to install dependencies"

# Cleaning targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	@echo "Cleaned build artifacts and cache files"

# Development targets
test:
	python -m pytest tests/ -v --cov=services --cov-report=html --cov-report=term

lint:
	flake8 services/ scripts/ tests/
	mypy services/ scripts/

format:
	black services/ scripts/ tests/
	isort services/ scripts/ tests/

# Demo and running targets
run-demo:
	python scripts/inference_examples.py

run-api:
	python scripts/api_server.py

train-sample:
	python scripts/train_classifier.py --create_sample --data_path ./data/sample_training.csv
	python scripts/train_classifier.py --data_path ./data/sample_training.csv --model_output_path ./models/sample_classifier.pkl

# Docker targets
docker-build:
	docker build -t smart-document-analyzer .

docker-run:
	docker run -p 8000:8000 smart-document-analyzer

# Development server
dev-server:
	uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000 --reload

# Model download (for CI/CD)
download-models:
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
	@echo "Models downloaded successfully"

# Quick start - complete setup
quickstart: setup install train-sample
	@echo ""
	@echo "✅ Quick start completed!"
	@echo ""
	@echo "Try these commands:"
	@echo "  make run-demo    # Run comprehensive demo"
	@echo "  make run-api     # Start API server at http://localhost:8000"
	@echo ""
	@echo "API Documentation will be available at:"
	@echo "  http://localhost:8000/docs"

# Check system requirements
check-system:
	@echo "Checking system requirements..."
	python --version
	pip --version
	@echo "Python path: $$(which python)"
	@echo "Available memory: $$(free -h 2>/dev/null || echo 'N/A (not Linux)')"
	@echo "System check completed"
