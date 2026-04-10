# Deep Semantic Search

A Python library for embedding, indexing, and applying semantic search for text and image data.

## Features

- **Multi-modal Semantic Search**
  - Embed and index text data using Sentence Transformers (nli-mpnet-base-v2)
  - Embed and index image data using CLIP
  - Search images by image or text queries
  - Search text by semantic similarity

- **Clustering & Captioning**
  - Cluster image embeddings using PyTorch KMeans (GPU support)
  - Caption images using BLIP
  - Customizable LLM-powered topic labeling via callback

- **Retrieval-Augmented Generation (RAG)**
  - Answer questions based on text data
  - Pluggable LLM via callback pattern

## Installation

```bash
pip install deep-semantic-search
```

For development:
```bash
pip install deep-semantic-search[dev]
```

## Quick Start

### Image Search

```python
from deep_semantic_search import LoadImageData, ImageIndexer, ImageSearcher

# Load images
loader = LoadImageData()
image_paths = loader.from_folder(["path/to/images"])

# Index images
indexer = ImageIndexer(image_paths)
indexer.run_index()

# Search by text
searcher = ImageSearcher(indexer)
results = searcher.search_by_text("cat on a sofa", n=5)
for path, score in results.items():
    print(f"{score:.3f}  {path}")

# Search by image
results = searcher.search_by_image("query.jpg", n=5)
```

### Text Search

```python
from deep_semantic_search import LoadTextData, TextEmbedder, TextSearch

# Load text data
loader = LoadTextData()
corpus = loader.from_folder("path/to/text/files")

# Embed
embedder = TextEmbedder()
embedder.embed(corpus)

# Search
search = TextSearch(embedder)
results = search.find_similar("your search query", top_n=5)
for r in results:
    print(f"Score: {r['score']:.3f}  {r['path']}")
```

### Image Clustering

```python
from deep_semantic_search import ImageIndexer, ImageClusterer, ImageCaptioner

indexer = ImageIndexer(image_paths)
indexer.run_index()

# Optional: use captioner for topic labels
captioner = ImageCaptioner()
clusterer = ImageClusterer(indexer)
result = clusterer.cluster(n_clusters=5, captioner=captioner)

# Save organized clusters to disk
clusterer.save_clusters("./output/clusters")
```

### RAG (Question Answering)

```python
from deep_semantic_search import ask_question

texts = ["Document 1 content...", "Document 2 content..."]
answer = ask_question(texts, "What is the main topic?")
print(answer)

# With a custom LLM
answer = ask_question(texts, "Summarize this.", llm_fn=my_custom_llm)
```

### Custom Data Paths

By default, metadata is stored in `~/.deep-semantic-search/`. Override per instance:

```python
indexer = ImageIndexer(image_paths, metadata_dir="./my_project/index")
embedder = TextEmbedder(metadata_dir="./my_project/text_index")
```

## API Reference

### Image Module
- `LoadImageData` — Load image paths from folders or CSV
- `ImageIndexer` — CLIP embedding + FAISS indexing
- `ImageSearcher` — Image/text similarity search
- `ImageClusterer` — KMeans clustering with topic labeling
- `ImageCaptioner` — BLIP image captioning

### Text Module
- `LoadTextData` — Load text from folders (.txt/.html) or CSV
- `TextEmbedder` — Sentence Transformer embeddings
- `TextSearch` — Cosine similarity search

### RAG
- `ask_question()` — RAG Q&A with pluggable LLM

### Exceptions
- `DeepSemanticSearchError` — Base exception
- `IndexNotFoundError`, `ModelLoadError`, `SearchError`, `EmbeddingError`, `ClusteringError`

## Requirements

- Python >= 3.10
- PyTorch, Sentence Transformers, Transformers, FAISS, LangChain, and more (auto-installed)

## License

MIT
# [Deep Semantic Search](https://github.com/Harduex/deep-semantic-search)

This repository contains a system designed for embedding, indexing, and applying semantic search for personal folders containing text and image data.<br>
The system is capable of processing, analyzing, and visualizing the data, with additional features such as clustering, image captioning, and retrieval-augmented generation.

## Components:

**Multi-modal [Semantic Search](https://en.wikipedia.org/wiki/Semantic_search)**:

- Embedding and indexing text data using the nli-mpnet-base-v2 model.
- Embedding and indexing image data using the CLIP model.
- Semantic search for both text and image data (searching images by both image and text queries).
- Additional keyword text search feature for enhanced search results.

**Clustering and Image Captioning**:

- Clustering image embeddings using the PyTorch KMeans implementation (with GPU support).
- Image captioning utilizing the BLIP model.

**Retrieval-Augmented Generation [RAG](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)**:

- Utilization of a local instance of the Ollama API to run open-source LLM models (running with docker-compose).
- Answering questions based on search results.
- Summarizing search results.
- Generating topics for provided image captions.

**Web User Interface Using [Gradio](https://gradio.app/)**:

- Provides a user-friendly interface for interacting with the system.

**Visualization (In experiments directory)**

- Visualizes data and results.
- Facilitates exploration of topic relationships through semantic graphs.
- Applies PCA dimensionality reduction for 2D and 3D visualizations of cluster embeddings.

**Backend API Support**:

- Offers a RESTful API for data retrieval and processing.

## Download the Example Testing Dataset:

A sample testing dataset can be downloaded from [here](https://drive.google.com/file/d/150JAF09H_Dg4Q-fzqmvhB1vJ3Nvf7RYr).

## Installation (Linux / MacOS):

*(Recommended)*

### Configuration:

```bash
cp .env.example .env
```

### Starting the System:

```bash
./start.sh
```

Access the web interface at [http://127.0.0.1:7860/](http://127.0.0.1:7860/).

### Running Tests:

```bash
python ./src/api.py
cd src/tests
pytest
```

## How to Run Manually (Windows):

Please note that the system is primarily designed to run on Linux. Running on Windows may require additional adjustments and is not guaranteed to work seamlessly.

```bash
# Set environment variables
set OLLAMA_LLM_MODEL=your_model # default is mistral:7b
set DEFAULT_SEARCH_FOLDER_PATH=\path\to\your\dataset\folder # optional

# Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Start Ollama API and pull the model
docker compose up -d
docker exec -it ollama-api ollama pull %OLLAMA_LLM_MODEL%

# Start the application
python .\src\app.py
```

Access the web interface at [http://127.0.0.1:7860/](http://127.0.0.1:7860/).
