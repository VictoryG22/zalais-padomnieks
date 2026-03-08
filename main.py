from fastapi import FastAPI
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from rag_pipeline import rag_answer

# -----------------------------
# Настройка
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ Не найден OPENAI_API_KEY или PINECONE_API_KEY в .env!")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------------
# Подключение к Pinecone
# -----------------------------
Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()
chat_history = []

class Question(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "Zaļais padomnieks API running 🌱"}

@app.post("/chat")
def chat(question: Question):
    result = rag_answer(
        question.message,
        chat_history,
        vectorstore
    )

    # обновляем историю
    chat_history.append({"role":"user","content":question.message})
    chat_history.append({"role":"assistant","content":result["answer"]})

    return result
