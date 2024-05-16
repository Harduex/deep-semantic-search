import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.schema import Document

QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL") or "mistral:7b"

def ask_question(text_data, question, chunk_size=1500, chunk_overlap=100, model_name=LLM_MODEL, prompt=QA_CHAIN_PROMPT):
    data = [Document(page_content=text) for text in text_data]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
    llm = Ollama(model=model_name, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": prompt})
    result = qa_chain({"query": question})
    return result['result']
