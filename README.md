# Multilingual Patent Search Engine

## Overview
This project implements a multilingual search engine for patent texts using modern NLP techniques. It can process and search through patent documents in multiple languages, utilizing sentence embeddings for semantic search capabilities.

### Key Features
- Multilingual support with automatic language detection
- Semantic search using sentence transformers
- Fast vector similarity search with Qdrant 
- 🚀 REST API interface using FastAPI
- Efficient data caching system

## Technical Stack
- **Python 3.11**
- **FastAPI** - Web framework for the API
- **Sentence-Transformers** - For text embeddings
- **Qdrant** - Vector similarity search ((The current implementation uses Qdrant's in-memory storage mode for development and testing))
- **LangDetect** - Language detection
- **Poetry** - Dependency management

## Getting Started

### Prerequisites
- Python 3.11+
- Poetry for dependency management

### Installation
```bash
# Clone the repository
git clone git@github.com:raoulvanrossum/text_challenge.git
cd text_challenge


# Optional: 
poetry env use python3.11

# Install dependencies
poetry install
```

### Running the Application
```bash
# Start the FastAPI server
poetry run uvicorn patent_search.api.app:app
```

## API Endpoints

### Upload Patent Text
```http
POST /api/patents/add
```
Upload new patent texts to the search engine.

### Search Patents
```http
POST /search
```
Search for similar patents using a query text.

### Parameters
- `text`: The search query text
- `k`: Number of results to return (default: 5)
- `threshold`: Similarity threshold (default: 0.5)

## Example Usage

### Searching Patents
Browse to http://127.0.0.1:8000/ and insert keywords. 


```python
import requests

payload = {
    "text": "quantum computing implementation",
    "k": 5
}
response = requests.post("http://localhost:8000/search", json=payload)
```

## Testing
Run the test suite:
```bash
poetry run pytest -v src
```

## Project Structure
```
patent-search/
├── src/
│   └── patent_search/
│       ├── api/
│       │   └── app.py       # API endpoints
│       ├── core/
│       │   ├── indexer.py   # Vector indexing and search
│       │   └── processor.py # Text processing
│       └── config.py        # Configuration settings
├── data/
│   ├── raw/
│   └── processed/
├── tests/
├── pyproject.toml
└── README.md
```

## Technical Details

### Text Processing Pipeline
1. Language Detection
2. Text Normalization
3. Embedding Generation
4. Vector Indexing

### Search Algorithm
- Uses cosine similarity for vector matching
- Supports multilingual queries
- Implements efficient vector search with Qdrant (currently stored in memory)

## Performance
- Handles multiple languages efficiently
- Fast search response times
- Scalable vector storage

## Explainability

The search engine implements several features that make its results transparent and explainable:

### 1. Semantic Similarity Scores
- Each search result includes a similarity score (0-1)
- Scores indicate how closely the result matches the query
- Higher scores (closer to 1) indicate stronger semantic matches

### 2. Language Detection Transparency
- The system automatically detects and reports the language of both query and patents
- Users can understand which languages are being processed
- Helps explain potential matches across different languages

### 3. Vector-based Matching Process
The search process is transparent and can be broken down into steps:
1. Text → Embeddings: Converts text to numerical vectors using sentence transformers
2. Similarity Computation: Uses cosine similarity for comparing vectors
3. Ranking: Results are ordered by similarity score

### 4. Result Interpretation
Each search result provides:
- The matched text
- Similarity score
- Language information
- Original metadata

## Embedding Model Selection

### Current Model
The system uses `intfloat/multilingual-e5-small` as the default embedding model. 
You are able to change this in the src/patent_search.config.py.

## Adding New Patents via API

You can add new patents to the system using the API endpoint `/api/patents/add`. Here's how to do it using curl:
The patent upload service is designed to be robust and non-blocking, implementing several reliability features:

#### 1. Background Processing
- Upload requests are processed asynchronously using FastAPI's background tasks
- The main upload process is non-blocking, allowing the API to remain responsive
- Cache updates happen in the background after the index is updated
### Basic Usage

```bash
curl -X POST http://localhost:8000/api/patents/add \
-H "Content-Type: application/json" \
-d '{
    "patents": [
        {
            "text": "Your patent text here",
            "metadata": {
                "source": "manual_upload",
                "category": "Technology"
            }
        }
    ]
}'
```
### How the Service Handles Uploads

The patent upload service is designed to be robust and non-blocking, implementing several reliability features:

#### 1. Background Processing
- Upload requests are processed asynchronously using FastAPI's background tasks
- The main upload process is non-blocking, allowing the API to remain responsive
- Cache updates happen in the background after the index is updated


## Limitations/improvements
- Requires sufficient memory for embedding operations (the system processes patent abstracts as complete units rather than individual words or sentences for this reson)
- Limited to languages supported by the underlying model
- Qdrant server (not memory)
