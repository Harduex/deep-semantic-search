### System Overview:
The system is designed to be a comprehensive note-taking and information management platform, leveraging advanced machine learning techniques to enhance user interaction and data organization. This system aims to provide a seamless and intelligent experience for managing and interacting with personal or professional notes and documents.

### Components:
**Semantic Search (Txtai)**:
   - Utilizes embeddings for text and image search.
   - Incorporates an additional keyword search feature.

**Open Source LLMs API (Ollama)**:
   - Reads, formats, and summarizes search results.
   - Answers questions based on the dataset.

**Data Management and Processing (Pandas, LangChain)**:
   - Supports various data formats including Google Keep exports, documents, and data frames.
   - Uses Retrieval-Augmented Generation (RAG) for processing.

**Automatic Text Processing (Scikit-Learn, PyTorch)**:
   - Features automatic labeling of text and zero-shot topic classification.
   - Employs t-SNE for thematic grouping of notes.
   - Utilizes Top2Vec for topic modelling.
   - Sentiment Analysis for each topic or note.

**Web Search Integration (LangChain Agents, Searx)**:
   - Uses the Searx engine and LangChain Agents for extended web searches.
   - Provides a daily random note/quote feature.

**Visualization Tools (Plotly, Matplotlib)**:
   - Generates visual representations like charts and word clouds.
   - Enables exploration of topic relationships through semantic graphs.
   - Visualization of changes in topics over time (Topic Evolution Tracking).

**Topics Suggestion and Summarization**:
   - Groups similar notes together and adds LLM-generated summaries.
   - Offers internet search integration for new and related topics suggestion.

**Web User Interface (Next.js, React)**:
   - Google Keep-inspired web UI with a simple search box and results display.
   - Allows for UI customization by forking existing designs.

**Backend API (Flask, Python)**:
   - Provides RESTful API for data retrieval and processing.
   - Supports data export and import functionalities.

**Deployment (Docker, Docker Compose)**:
    - Packages the application and services using Docker.

### Run with docker:
```bash
./start.docker.sh
```

### Local development:
```bash
./start.local.sh
```

### Example .env file:
```bash
OLLAMA_LLM_MODEL=mistral:7b
```