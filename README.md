## Team AfriTrade
### slack team
- jeffdauda
- jaboprosper
- moussamoustapha
- ssmoses
  
## About repo
This FastAPI-based project creates a Retrieval-Augmented Generation (RAG) system that answers legal questions related to the transportation of goods across different African countries. The system processes legal documents in PDF format, retrieves relevant information, and generates responses using OpenAI's GPT models.

Model deployment link: https://af-rag.onrender.com/docs

## Features

PDF Document Loading: Automatically loads and processes all PDF files from a specified folder.
Document Splitting: Splits documents into manageable chunks using recursive text splitting.
FAISS Vector Store: Initializes a FAISS vector store for efficient document retrieval based on embeddings.
- Retrieval-Augmented Generation: Combines document retrieval with GPT-3.5-turbo from OpenAI to generate accurate responses based on the context.
 CORS Support: Configured for integration with a frontend.

## Tech Stack

- FastAPI: For creating the API.
- LangChain: For document processing, retrieval, and OpenAI API integration.
- OpenAI: GPT-3.5-turbo model for generating legal responses.
- FAISS: Vector database for fast retrieval of document embeddings.
- PyPDFLoader: For loading and reading PDF files.
