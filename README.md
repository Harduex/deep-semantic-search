# Deep Semantic Search
This system is designed for embedding, indexing and applying semantic search for personal folders with any text and image data inside. <br>
The system is able to process, analyze and visualize the data. The user can interact with the system via web user interface.

## Components:
**Multi-modal [Semantic Search](https://en.wikipedia.org/wiki/Semantic_search) (Custom [DeepImageSearch](https://github.com/TechyNilesh/DeepImageSearch) and [DeepTextSearch](https://github.com/TechyNilesh/DeepTextSearch), [CLIP](https://openai.com/research/clip), [nli-mpnet-base-v2](https://huggingface.co/sentence-transformers/nli-mpnet-base-v2))**:
   - Embedding and indexing the text data using the nli-mpnet-base-v2 model.
   - Embedding and indexing the image data using the CLIP model.
   - Semantic search for text and image data (searching images by both image and text as a query).
   - Additional keyword text search feature for better search results. 

**Clustering and Image Captioning ([KMeans](https://github.com/subhadarship/kmeans_pytorch), [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large))**:
   - Clustering the image embeddings using PyTorch KMeans implementation (for a GPU support).
   - Image captioning using BLIP model.

**Retrieval-Augmented Generation [RAG](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) ([Ollama](https://github.com/ollama/ollama), [Docker](https://www.docker.com/))**:
   - Using local instance of Ollama API to run open-source LLM models. (running with docker-compose)
   - Answering questions based on the search results.
   - Summarizing the search results.
   - Picking topics for provided image captions.

**Web User Interface ([Gradio](https://www.gradio.app/))**:
   - Allows the user to interact with the system.

**Visualization Tools ([Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/))**:
   - Visualizing the data and the results.
   - Enables exploration of topic relationships through semantic graphs.
   - Applying PCA dimensionality reduction for 2D and 3D visualizations of the clusters embeddings.

**Backend API support ([Flask](https://github.com/pallets/flask))**:
   - Provides RESTful API for data retrieval and processing.
   - Supports data export and import functionalities.

## Download the example testing dataset from here:
https://drive.google.com/file/d/150JAF09H_Dg4Q-fzqmvhB1vJ3Nvf7RYr

## Installation (Linux / MacOS)
(Recommended)

### Configuration
```bash
cp .env.example .env
```

### Start the system
```bash
./start.sh
```
Access the web interface on http://127.0.0.1:7860/

### Run tests
```bash
python ./src/api.py
cd src/tests
pytest
```

## How to run manually (Windows)
Keep in mind that the system is designed to run on Linux. 
The system is not guaranteed to work on Windows and may require additional tweaks.
```
# Set environment variables
set OLLAMA_LLM_MODEL=your_model # default is mistral:7b
set DEFAULT_SEARCH_FOLDER_PATH=\path\to\your\dataset\folder # optional

# Create a virtual environment and install the dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Start Ollama API and pull the model
docker compose up -d
docker exec -it ollama-api ollama pull %OLLAMA_LLM_MODEL%

# Start the application
python .\src\app.py
```
Access the web interface on http://127.0.0.1:7860/