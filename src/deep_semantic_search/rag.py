"""Retrieval-Augmented Generation (RAG) for question answering."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from .exceptions import SearchError

logger = logging.getLogger("deep_semantic_search")

DEFAULT_RAG_PROMPT_TEMPLATE = (
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Use three sentences maximum and keep the answer as concise as possible.\n\n"
    "{context}\n\nQuestion: {question}\n\nHelpful Answer:"
)


def _get_default_prompt():
    """Try to pull the RAG prompt from LangChain Hub; fall back to built-in."""
    try:
        from langchain import hub

        return hub.pull("rlm/rag-prompt-llama")
    except Exception:
        from langchain_core.prompts import PromptTemplate

        return PromptTemplate(
            input_variables=["context", "question"],
            template=DEFAULT_RAG_PROMPT_TEMPLATE,
        )


def _get_default_llm(model_name: str):
    """Create a default Ollama LLM instance."""
    from langchain_community.llms import Ollama

    return Ollama(model=model_name, verbose=False)


def ask_question(
    text_data: list[str],
    question: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    llm_fn: Callable[[str], str] | None = None,
    model_name: str | None = None,
    prompt=None,
) -> str:
    """Answer a question using RAG over the provided text data.

    Parameters
    ----------
    text_data : list[str]
        List of text documents to search over.
    question : str
        The question to answer.
    chunk_size : int
        Size of text chunks for splitting.
    chunk_overlap : int
        Overlap between chunks.
    llm_fn : Callable[[str], str] | None
        Optional custom LLM callable. If provided, ``model_name`` is ignored
        and this function is used directly for generation.
    model_name : str | None
        Ollama model name. Defaults to env var ``OLLAMA_LLM_MODEL`` or ``mistral:7b``.
    prompt
        LangChain prompt template. Defaults to built-in RAG prompt.

    Returns
    -------
    str
        The generated answer.
    """
    if not text_data:
        raise SearchError("text_data must be a non-empty list of strings.")

    if model_name is None:
        model_name = os.getenv("OLLAMA_LLM_MODEL") or "mistral:7b"

    if prompt is None:
        prompt = _get_default_prompt()

    try:
        from langchain_community.embeddings import GPT4AllEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        documents = [Document(page_content=text) for text in text_data]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=GPT4AllEmbeddings()
        )

        if llm_fn is not None:
            from langchain_core.language_models import BaseLLM

            class _CallableLLM(BaseLLM):
                _fn: Callable[[str], str]

                def __init__(self, fn: Callable[[str], str], **kwargs):
                    super().__init__(**kwargs)
                    self._fn = fn

                def _call(self, prompt: str, **kwargs) -> str:
                    return self._fn(prompt)

                def _generate(self, prompts, stop=None, **kwargs):
                    from langchain_core.outputs import Generation, LLMResult
                    generations = [[Generation(text=self._fn(p))] for p in prompts]
                    return LLMResult(generations=generations)

                @property
                def _llm_type(self) -> str:
                    return "custom_callable"

            llm = _CallableLLM(fn=llm_fn)
        else:
            llm = _get_default_llm(model_name)

        retriever = vectorstore.as_retriever()

        def _format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)

    except SearchError:
        raise
    except Exception as exc:
        raise SearchError(f"RAG question answering failed: {exc}") from exc
