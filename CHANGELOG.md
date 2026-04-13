# Changelog

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
