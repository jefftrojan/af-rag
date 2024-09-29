from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your React app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to load and process PDF documents
def load_and_process_pdfs(pdf_folder_path):
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits

# Function to initialize the vector store with documents
def initialize_vectorstore(splits):
    return FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=api_key))

pdf_folder_path = "./docs"
splits = load_and_process_pdfs(pdf_folder_path)
vectorstore = initialize_vectorstore(splits)

prompt_template = """You are a legal expert. You need to answer the questions related to legal requiremeents to transport goods/products across different countries in  Africa. 
Given below is the context and question of the user. Don't answer question outside the context provided. 
context = {context}
question = {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class UserInput(BaseModel):
    user_input: str

@app.post("/rag")
async def generate_response(user_input: UserInput):
    try:
        response = rag_chain.invoke(user_input.user_input)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8501))
    uvicorn.run(app, host="localhost", port=port)