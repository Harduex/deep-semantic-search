# Changelog

## 3.0.3

### Bug Fixes

- **Fix unified search returning only text results**: Scores are now normalized per-modality so image and text results are ranked on a comparable scale. Previously, text embeddings dominated due to SigLIP's 64-token limit inflating text similarity scores.

## 3.0.2

### Bug Fixes

- **Pin `transformers>=4.38.0,<4.54.0`**: Florence-2 custom modeling code is incompatible with transformers >= 4.54.0 (`_supports_sdpa` attribute error).
- **Fix HDBSCAN clustering for small datasets**: Default `min_cluster_size` lowered from 5 to 3, auto-adjusted for dataset size, and `min_samples=1` prevents all-noise results on small image sets.

## 3.0.1

### Bug Fixes

- **Pin `transformers>=4.38.0,<5.0.0`**: FlagEmbedding and Florence-2 are incompatible with transformers 5.x. This pin ensures all features work correctly.
- **Add `einops` and `timm` to `[captioning]` extra**: Florence-2 requires these packages but they were missing from the optional dependency list.
- **Fix misleading install instructions**: Reranking error messages no longer point to the empty `[reranking]` extra; they correctly reference the base package.
- **Fix CI workflow branch target**: CI now runs on `master` (was incorrectly targeting `main`).

## 3.0.0

### Breaking Changes

- **Image model: CLIP → SigLIP SO400M** (`google/siglip-so400m-patch14-384`, 1152-dim, 384×384 input). All image embeddings must be rebuilt.
- **Text model: MiniLM-L12 → BGE-M3** (`BAAI/bge-m3`, 1024-dim dense + sparse vectors). All text embeddings must be rebuilt.
- **Vector backend: FAISS → USearch**. Index files change from `.idx` to `.usearch`. Legacy FAISS indexes are auto-migrated from existing `.npy` features.
- **Captioning model: BLIP → Florence-2** (`microsoft/Florence-2-large`). `ImageCaptioner.caption()` no longer accepts `starting_text`; uses `task` parameter instead (default `"<DETAILED_CAPTION>"`). Model loading is now lazy.
- **RAG stack: LangChain → LiteLLM + USearch**. All `langchain-*` dependencies removed. Install `pip install deep-semantic-search[llm]` for RAG.
- **`TextEmbedder.load_embedding()` returns 3-tuple** `(dense_ndarray, sparse_list_or_None, corpus_dict)` instead of `(tensor, dict)`.
- **`ImageClusterer.cluster()` `n_clusters` is now optional**. Omit for HDBSCAN auto-clustering; provide for KMeans.
- **`search_by_text()` scores are now cosine similarity** in [0, 1] range (was unbounded inner product).

### New Features

- **Hybrid search**: BGE-M3 native dense + sparse vector fusion via weighted sum. Enable with `hybrid=True` (default) in `TextSearch.find_similar()`.
- **Cross-encoder reranking**: `rerank=True` flag on `TextSearch.find_similar()` and RAG. Uses `BAAI/bge-reranker-v2-m3`.
- **Cross-modal unified search**: `UnifiedIndexer` + `UnifiedSearcher` — index images and text in a shared SigLIP embedding space, search across modalities.
- **Duplicate detection**: `find_duplicates(threshold)` method on `ImageSearcher`, `TextSearch`, and `UnifiedSearcher`.
- **Semantic chunking**: RAG now splits text at embedding-similarity boundaries instead of fixed character counts. Toggle with `semantic_chunking=True/False`.
- **`RAG` class**: Object-oriented API alongside the existing `ask_question()` wrapper.
- **New CLI commands**: `unified-search`, `find-duplicates`.
- **CLI enhancements**: `text-search` gains `--rerank` and `--hybrid/--no-hybrid`; `image-cluster` gains optional `-k` (omit for HDBSCAN) and `--min-cluster-size`; `ask` gains `--rerank` and `--semantic-chunking/--no-semantic-chunking`.

### Dependencies

- **Removed**: `faiss-cpu`, all `langchain-*` packages
- **Added (core)**: `usearch>=2.9.0`, `FlagEmbedding>=1.2.0`
- **New optional extras**: `[llm]` (litellm), `[captioning]`, `[reranking]`

### Migration Guide

```python
# v2: text embeddings
embeddings, corpus = embedder.load_embedding()

# v3: now returns sparse vectors too
dense, sparse, corpus = embedder.load_embedding()

# v2: clustering required n_clusters
clusterer.cluster(n_clusters=5)

# v3: omit for HDBSCAN auto-detection
clusterer.cluster()  # HDBSCAN
clusterer.cluster(n_clusters=5)  # KMeans

# v2: RAG with LangChain
from deep_semantic_search import ask_question
answer = ask_question(texts, "question")

# v3: RAG with LiteLLM (install deep-semantic-search[llm])
from deep_semantic_search import RAG
rag = RAG(rerank=True)
answer = rag.ask(texts, "question")
# or use backward-compat wrapper:
answer = ask_question(texts, "question", llm_fn=my_fn)
```

## 2.0.0

### Breaking Changes

- **`ImageSearcher.search_by_image()` and `search_by_text()` return type changed** from `dict[str, float]` to `list[dict]` with keys `rank`, `path`, `score`. This matches the existing `TextSearch.find_similar()` return format.
- **`kmeans-pytorch` replaced with `scikit-learn`** for image clustering. The `ImageClusterer.cluster()` method no longer uses GPU-accelerated KMeans. Install with `pip install deep-semantic-search[clustering]`.
- **Serialization format changed** from pickle to numpy `.npy` + JSON. Existing pickle files are auto-migrated on first load (old files preserved as backup).
- **Heavy dependencies are now optional extras:**
  - `pip install deep-semantic-search[rag]` for RAG/question answering (langchain, chromadb)
  - `pip install deep-semantic-search[clustering]` for image clustering (scikit-learn)
  - `pip install deep-semantic-search[viz]` for plotting (matplotlib)
  - `pip install deep-semantic-search[all]` for everything

### Improvements

- **FAISS index cached in memory** -- no longer re-read from disk on every search call
- **Lazy model loading** -- CLIP and SentenceTransformer models are loaded on first use, not during construction
- **Deprecated LangChain imports fixed** -- uses `langchain_core.messages`, `langchain_chroma`, `langchain_ollama`
- **Deprecated `chat()` call replaced** with `chat.invoke()` in image clusterer
- **Pickle deserialization removed** -- embeddings stored as `.npy`, metadata as JSON (eliminates arbitrary code execution risk)
- **Device detection centralized** in `config.get_device()`
- **Ollama model default consolidated** into `config.DEFAULT_OLLAMA_MODEL`
- **`LoadTextData` stateful bug fixed** -- `corpus_dict` no longer accumulates across multiple `from_*` calls
- **`.data.numpy()` replaced** with `.detach().numpy()` in text searcher
- **Dead `_call()` method removed** from RAG's internal `_CallableLLM` class
- **Python 3.13 classifier added**

### Migration Guide

```python
# v1.x
results = searcher.search_by_text("query", n=5)
for path, score in results.items():
    print(f"{score:.3f}  {path}")

# v2.0
results = searcher.search_by_text("query", n=5)
for r in results:
    print(f"{r['score']:.3f}  {r['path']}")
```

## 1.1.4

- Use consistent text model (paraphrase-multilingual-MiniLM-L12-v2) for RAG embeddings
- Update .env default to gemma4:e4b

## 1.1.3

- Replace GPT4AllEmbeddings with HuggingFaceEmbeddings (no gpt4all dependency needed)
- Replace deprecated langchain_community Ollama with langchain_ollama
- Add langchain-huggingface, langchain-ollama, langchain-text-splitters, chromadb deps

## 1.1.2

- Change default Ollama model to gemma4:e4b

## 1.1.1

- Fix transformers 5.x compatibility (pooler_output check)
- Lint cleanup

## 1.1.0

- Add CLI tool (`dss`) with image-search, text-search, image-cluster, ask commands
- Add demo notebook with sample data
- Change default text model to paraphrase-multilingual-MiniLM-L12-v2

## 1.0.0

- Initial pip package release
- Refactored from monolithic codebase into clean library structure
