import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote

from tqdm import tqdm
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone

from dotenv import load_dotenv
import os

# --------------------------------
# CONFIG
# --------------------------------

BASE_URL = "https://biofruitnet.eu/resources"
PDF_DIR = "data/pdfs"

PINECONE_INDEX = "zalais-padomnieks"

os.makedirs(PDF_DIR, exist_ok=True)

load_dotenv()  # загружает .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ Не найден OPENAI_API_KEY! Проверь .env файл")

# для LangChain/OpenAI client
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --------------------------------
# SCRAPE PDF LINKS
# --------------------------------

def extract_pdf_links():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36"
    }

    try:
        r = requests.get(BASE_URL, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print("❌ Ошибка соединения:", e)
        return []

    soup = BeautifulSoup(r.text, "lxml")

    links = []

    for a in soup.select("a[href]"):
        href = a.get("href")
        if href and ".pdf" in href.lower():
            full = urljoin(BASE_URL, href)
            links.append(full)

    return list(set(links))

# --------------------------------
# DOWNLOAD PDF
# --------------------------------

def filename_from_url(url):
    return unquote(urlparse(url).path.split("/")[-1])

def download_pdfs(urls):

    files = []

    for url in tqdm(urls):

        name = filename_from_url(url)
        path = os.path.join(PDF_DIR, name)

        if os.path.exists(path):
            files.append(path)
            continue

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/114.0.0.0 Safari/537.36"
        }

        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print("❌ Ошибка соединения:", e)
            urls = []
        else:
            soup = BeautifulSoup(r.text, "lxml")
            urls = [a["href"] for a in soup.select("a[href$='.pdf']")]
            print(f"Найдено PDF: {len(urls)}")

        with open(path, "wb") as f:
            f.write(r.content)

        files.append(path)

    return files

# --------------------------------
# PDF → CHUNKS
# --------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)

def pdf_to_docs(pdf_path):

    reader = PdfReader(pdf_path)

    docs = []

    for i, page in enumerate(reader.pages):

        text = page.extract_text()

        if not text:
            continue

        chunks = splitter.split_text(text)

        for j, ch in enumerate(chunks):

            docs.append(
                Document(
                    page_content=ch,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": i+1
                    }
                )
            )

    return docs

# --------------------------------
# LOAD TO PINECONE
# --------------------------------

def upload_to_pinecone(docs):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
    )

    vectorstore = PineconeVectorStore.from_documents(
        docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX
    )

    print("Upload complete")

# --------------------------------
# MAIN
# --------------------------------

def main():

    print("Finding PDFs...")

    urls = extract_pdf_links()

    print("Found:", len(urls))

    pdfs = download_pdfs(urls)

    print("Processing PDFs...")

    all_docs = []

    for p in tqdm(pdfs):
        all_docs.extend(pdf_to_docs(p))

    print("Chunks:", len(all_docs))

    upload_to_pinecone(all_docs)


if __name__ == "__main__":
    main()
