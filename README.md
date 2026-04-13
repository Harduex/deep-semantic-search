# Deep Semantic Search

A Python library for embedding, indexing, and applying semantic search for text and image data.

## Features

- **Multi-modal Semantic Search**
  - Embed and index text data using Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
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

## CLI Tool

The package includes `dss`, a command-line interface for all major features. After installing the package, the `dss` command is available globally.

### General Usage

```bash
dss --help          # Show all commands
dss --version       # Show version
dss <command> --help  # Help for a specific command
```

Global flags: `-v`/`--verbose` for debug output, `-q`/`--quiet` to suppress progress.

### Image Search

Search images by text query or by image similarity:

```bash
# Search by text
dss image-search --folder ./photos --query "sunset over the ocean" --top 5

# Search by image
dss image-search --folder ./photos --query ./photos/reference.jpg --top 10

# Multiple folders, JSON output
dss image-search -f ./photos -f ./vacation --query "mountains" --format json

# Force re-indexing
dss image-search -f ./photos --query "cat" --reindex
```

### Text Search

Search text documents by semantic similarity:

```bash
dss text-search --folder ./documents "machine learning algorithms" --top 5

# CSV output
dss text-search -f ./docs "neural networks" --format csv

# Custom model
dss text-search -f ./docs "query" --model sentence-transformers/all-MiniLM-L6-v2
```

### Image Clustering

Cluster images using KMeans on CLIP embeddings:

```bash
# Basic clustering
dss image-cluster --folder ./photos --clusters 5

# With BLIP captioning for topic labels
dss image-cluster -f ./photos -k 5 --caption

# Save clustered images into organized folders
dss image-cluster -f ./photos -k 8 --caption --save-dir ./output/clusters

# JSON output
dss image-cluster -f ./photos -k 3 --format json
```

### RAG (Question Answering)

Ask questions over text documents using Retrieval-Augmented Generation:

```bash
dss ask --folder ./documents "What is the main conclusion?"

# Custom Ollama model
dss ask -f ./research "Summarize the findings" --model llama2:13b

# Adjust chunking
dss ask -f ./docs "question" --chunk-size 2000 --chunk-overlap 200
```

### Configuration

The CLI respects environment variables:
- `OLLAMA_LLM_MODEL` — LLM model for RAG (default: `mistral:7b`)
- `DEFAULT_SEARCH_FOLDER_PATH` — Default folder path

All CLI flags override environment variables when provided.

## Requirements

- Python >= 3.10
- PyTorch, Sentence Transformers, Transformers, FAISS, LangChain, and more (auto-installed)

## License

MIT
