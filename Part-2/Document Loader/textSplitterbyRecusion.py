import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

data=PyPDFLoader(r"C:\Users\Asus\Documents\t\G_AI\Part-2\Document Loader\GRU.pdf")

docs=data.load()

chunks=splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content)