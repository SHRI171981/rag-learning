# RAG Learning: LangChain with Gemini

## Problem statement

Large language models do not know your documents, and they answer in free-form text that programs cannot rely on. Building reliable AI apps means learning the pieces that fix this: prompts, chains, structured output, embeddings and retrieval.

## Solution

A set of hands-on scripts that build up the retrieval-augmented generation (RAG) toolkit one step at a time, using LangChain with Google Gemini models. The last step puts retrieval and generation together: documents are embedded, the closest ones to a question are found by cosine similarity, and the model answers from them.

## Tech stack

Python, LangChain, Google Gemini (chat and embedding models), Hugging Face Inference API, scikit-learn, python-dotenv.

## Features

- **Models:** chat models through Gemini and the Hugging Face API, and plain LLM calls
- **Prompts:** prompt templates and saved question-answering prompts
- **Chains:** prompt, model and output parser composed into a pipeline, with a printed graph
- **Structured output** three ways: Pydantic models, TypedDict and JSON schema
- **Chatbots:** a simple chatbot, a chat-prompt template and a full chatbot that keeps the conversation history
- **Embeddings:** Gemini and Hugging Face embedding models
- **Semantic search over documents:** embed documents, rank them against a query by cosine similarity, and have the model answer from the best matches

## Future steps

- Load and split real documents (PDFs, web pages)
- Store embeddings in a vector database such as FAISS or Chroma
- Build a complete retrieval chain with source citations
- Evaluate answer quality with a test set of questions

## Getting started

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key and run any script, for example:

```bash
python LangchainModels/EmbeddingModels/document_similarity.py
```
