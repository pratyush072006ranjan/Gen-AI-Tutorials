
#
# 1. Load PDF / TXT files
# 2. Split documents into chunks
# 3. Create embeddings
# 4. Store embeddings in ChromaDB


import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_mistralai import (
    MistralAIEmbeddings
)

from langchain_community.vectorstores import (
    Chroma
)



load_dotenv()


BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_DIR = BASE_DIR / "Document Loader"

CHROMA_DIR = BASE_DIR / "chroma-db"




if not DOCUMENT_DIR.exists():

    raise FileNotFoundError(
        f"Document Loader folder not found:\n{DOCUMENT_DIR}"
    )


print("=" * 60)
print("📚 DOCUMENT DATABASE CREATOR")
print("=" * 60)

print(
    f"\n📂 Document folder:\n{DOCUMENT_DIR}"
)


documents = []




pdf_files = list(
    DOCUMENT_DIR.glob("*.pdf")
)

for pdf_file in pdf_files:

    print(
        f"\n📄 Loading PDF: {pdf_file.name}"
    )

    loader = PyPDFLoader(
        str(pdf_file)
    )

    docs = loader.load()

    documents.extend(docs)



txt_files = list(
    DOCUMENT_DIR.glob("*.txt")
)

for txt_file in txt_files:

    print(
        f"\n📝 Loading TXT: {txt_file.name}"
    )

    loader = TextLoader(
        str(txt_file),
        encoding="utf-8"
    )

    docs = loader.load()

    documents.extend(docs)


if not documents:

    raise ValueError(
        "No PDF or TXT files were found inside "
        "'Document Loader'."
    )


print(
    f"\n Total loaded documents/pages: {len(documents)}"
)


print(
    "\n Splitting documents into chunks..."
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    documents
)

print(
    f" Total chunks created: {len(chunks)}"
)

print(
    "\n Creating Mistral embedding model..."
)

embedding_model = MistralAIEmbeddings()



print(
    "\n Creating Chroma vector database..."
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(CHROMA_DIR)
)


print("\n" + "=" * 60)
print(" VECTOR DATABASE CREATED SUCCESSFULLY")
print("=" * 60)

print(
    f"\n ChromaDB location:\n{CHROMA_DIR}"
)

print(
    f"\n Documents/pages: {len(documents)}"
)

print(
    f" Chunks: {len(chunks)}"
)

print("\n Your RAG system is ready!")