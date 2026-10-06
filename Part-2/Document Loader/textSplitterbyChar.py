import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter=CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

data=TextLoader(r"C:\Users\Asus\Documents\t\G_AI\Part-2\Document Loader\notes2.txt")

docs=data.load()
chunks=splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()