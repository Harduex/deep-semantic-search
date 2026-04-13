# Deep Semantic Search

A Python library for embedding, indexing, and applying semantic search for text and image data.

## Features

- **Multi-modal Semantic Search**
  - Embed and index images using SigLIP SO400M (1152-dim, 384×384)
  - Embed and index text using BGE-M3 (1024-dim dense + sparse vectors)
  - Search images by image or text queries
  - Search text by semantic similarity with hybrid dense+sparse fusion
  - Cross-modal unified search across images and text in a shared embedding space

- **Clustering & Captioning**
  - Cluster image embeddings using KMeans (specify k) or HDBSCAN (auto-detect)
  - Caption images using Florence-2 (detailed captions, object detection, OCR)
  - Customizable LLM-powered topic labeling via callback

- **Retrieval-Augmented Generation (RAG)**
  - Answer questions based on text data using LiteLLM + Ollama
  - Semantic chunking with BGE-M3 embeddings
  - Cross-encoder reranking with BGE-reranker-v2-m3
  - Pluggable LLM via callback pattern

- **Duplicate Detection**
  - Find near-duplicate images or text above a similarity threshold

## Installation

```bash
pip install deep-semantic-search
```

Install with optional extras:
```bash
pip install deep-semantic-search[llm]          # RAG / question answering (LiteLLM)
pip install deep-semantic-search[clustering]   # Image clustering (scikit-learn)
pip install deep-semantic-search[viz]          # Plotting / visualization
pip install deep-semantic-search[all]          # Everything
```

For development:
```bash
pip install deep-semantic-search[dev]
```

## Quick Start

### Image Search

```python
from deep_semantic_search import LoadImageData, ImageIndexer, ImageSearcher

# Load and index images
loader = LoadImageData()
image_paths = loader.from_folder(["path/to/images"])

indexer = ImageIndexer(image_paths)
indexer.run_index()

# Search by text
searcher = ImageSearcher(indexer)
results = searcher.search_by_text("cat on a sofa", n=5)
for r in results:
    print(f"{r['score']:.3f}  {r['path']}")

# Search by image
results = searcher.search_by_image("query.jpg", n=5)

# Find duplicate images
duplicates = searcher.find_duplicates(threshold=0.95)
```

### Text Search

```python
from deep_semantic_search import LoadTextData, TextEmbedder, TextSearch

# Load and embed text data
loader = LoadTextData()
corpus = loader.from_folder("path/to/text/files")

embedder = TextEmbedder()
embedder.embed(corpus)

# Search with hybrid dense+sparse fusion
search = TextSearch(embedder)
results = search.find_similar("your search query", top_n=5, hybrid=True)

# With cross-encoder reranking
results = search.find_similar("query", top_n=5, rerank=True)
```

### Unified Cross-Modal Search

```python
from deep_semantic_search import UnifiedIndexer, UnifiedSearcher

# Index images and texts in a shared embedding space
indexer = UnifiedIndexer()
indexer.add_images(image_paths)
indexer.add_texts(["description 1", "description 2"], labels=["doc1", "doc2"])
indexer.build_index()

# Search across modalities
searcher = UnifiedSearcher(indexer)
results = searcher.search("sunset over mountains", n=10)
# Filter by modality
results = searcher.search("sunset", modality_filter="image")
```

### Image Clustering

```python
from deep_semantic_search import ImageIndexer, ImageClusterer, ImageCaptioner

indexer = ImageIndexer(image_paths)
indexer.run_index()

# Auto-detect clusters with HDBSCAN
clusterer = ImageClusterer(indexer)
result = clusterer.cluster()  # n_clusters=None → HDBSCAN

# Or specify exact number with KMeans
result = clusterer.cluster(n_clusters=5)

# With Florence-2 captioning for topic labels
captioner = ImageCaptioner()
result = clusterer.cluster(n_clusters=5, captioner=captioner)

# Save organized clusters to disk
clusterer.save_clusters("./output/clusters")
```

### RAG (Question Answering)

Requires `pip install deep-semantic-search[llm]` for LiteLLM.

```python
from deep_semantic_search import RAG

texts = ["Document 1 content...", "Document 2 content..."]

# With semantic chunking and reranking
rag = RAG(rerank=True)
answer = rag.ask(texts, "What is the main topic?", semantic_chunking=True)

# With a custom LLM
answer = rag.ask(texts, "Summarize this.", llm_fn=my_custom_llm)

# Backward-compatible wrapper
from deep_semantic_search import ask_question
answer = ask_question(texts, "What is the main topic?", llm_fn=my_fn)
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
- `ImageIndexer` — SigLIP embedding + USearch indexing
- `ImageSearcher` — Image/text similarity search + duplicate detection
- `ImageClusterer` — KMeans/HDBSCAN clustering with topic labeling
- `ImageCaptioner` — Florence-2 image captioning

### Text Module
- `LoadTextData` — Load text from folders (.txt/.html) or CSV
- `TextEmbedder` — BGE-M3 dense + sparse embeddings
- `TextSearch` — Hybrid search with optional reranking + duplicate detection

### Unified Search
- `UnifiedIndexer` — Cross-modal SigLIP indexing for images + text
- `UnifiedSearcher` — Search across modalities

### RAG
- `RAG` — Object-oriented RAG with semantic chunking and reranking
- `ask_question()` — Backward-compatible wrapper

### Exceptions
- `DeepSemanticSearchError` — Base exception
- `IndexNotFoundError`, `ModelLoadError`, `SearchError`, `EmbeddingError`, `ClusteringError`, `MigrationError`, `CaptioningError`

## CLI Tool

The package includes `dss`, a command-line interface for all major features.

### General Usage

```bash
dss --help              # Show all commands
dss --version           # Show version
dss <command> --help    # Help for a specific command
```

Global flags: `-v`/`--verbose` for debug output, `-q`/`--quiet` to suppress progress.

### Image Search

```bash
# Search by text
dss image-search --folder ./photos --query "sunset over the ocean" --top 5

# Search by image
dss image-search --folder ./photos --query ./photos/reference.jpg --top 10

# Multiple folders, JSON output
dss image-search -f ./photos -f ./vacation --query "mountains" --format json
```

### Text Search

```bash
# Basic search (hybrid enabled by default)
dss text-search --folder ./documents "machine learning algorithms" --top 5

# With reranking
dss text-search -f ./docs "neural networks" --rerank

# Dense-only (no sparse fusion)
dss text-search -f ./docs "query" --no-hybrid
```

### Image Clustering

```bash
# KMeans with explicit k
dss image-cluster --folder ./photos --clusters 5

# HDBSCAN auto-detection (omit -k)
dss image-cluster -f ./photos --min-cluster-size 3

# With Florence-2 captioning for topic labels
dss image-cluster -f ./photos -k 5 --caption

# Save clustered images
dss image-cluster -f ./photos -k 8 --save-dir ./output/clusters
```

### Unified Search

```bash
# Search across images and text
dss unified-search --image-folder ./photos --text-folder ./docs --query "sunset"

# Filter by modality
dss unified-search --image-folder ./photos --query "sunset" --filter image
```

### Duplicate Detection

```bash
dss find-duplicates --folder ./photos --threshold 0.95
```

### RAG (Question Answering)

```bash
dss ask --folder ./documents "What is the main conclusion?"

# With reranking and semantic chunking (default)
dss ask -f ./docs "Summarize the findings" --rerank

# Fixed chunking
dss ask -f ./docs "question" --no-semantic-chunking
```

### Configuration

The CLI respects environment variables:
- `OLLAMA_LLM_MODEL` — LLM model for RAG (default: `gemma4:e4b`)

## Requirements

- Python >= 3.10
- PyTorch, Sentence Transformers, Transformers, USearch, FlagEmbedding, and more (auto-installed)

## License

MIT
