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
