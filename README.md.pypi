# Deep Semantic Search

A Python library for embedding, indexing, and applying semantic search for text and image data.

## Features

- **Multi-modal Semantic Search**:
  - Embedding and indexing text data using the nli-mpnet-base-v2 model
  - Embedding and indexing image data using the CLIP model
  - Semantic search for both text and image data
  - Search images by both image and text queries

- **Clustering and Image Captioning**:
  - Cluster image embeddings using PyTorch KMeans (with GPU support)
  - Caption images using the BLIP model

- **Retrieval-Augmented Generation (RAG)**:
  - Answer questions based on search results
  - Summarize search results
  - Generate topics for image captions

## Installation

```bash
pip install deep-semantic-search
```

## Quick Start

### Text Search

```python
from deep_semantic_search import LoadTextData, TextEmbedder, TextSearch

# Load text data
loader = LoadTextData()
corpus_dict = loader.from_folder("path/to/text/files")

# Embed the text data
embedder = TextEmbedder()
embedder.embed(corpus_dict)

# Search for similar texts
search = TextSearch()
results = search.find_similar("your search query", top_n=5)

for result in results:
    print(f"Score: {result['score']}, Text: {result['text'][:100]}...")
```

### Image Search

```python
from deep_semantic_search import LoadImageData, ImageSearch

# Load image data
loader = LoadImageData()
image_paths = loader.from_folder("path/to/images")

# Set up image search
searcher = ImageSearch(image_paths)

# Search for similar images to a text query
results = searcher.get_similar_images_to_text("cat on a sofa", number_of_images=5)

# Display results
for path, score in results.items():
    print(f"Score: {score}, Image: {path}")
```

### RAG (Retrieval-Augmented Generation)

```python
from deep_semantic_search import ask_question

# Ask a question based on provided text data
texts = ["Text document 1...", "Text document 2..."]
answer = ask_question(texts, "What is the main topic discussed?")
print(answer)
```

## Requirements

- Python 3.8+
- PyTorch
- Sentence Transformers
- Hugging Face Transformers
- FAISS
- LangChain

## License

MIT