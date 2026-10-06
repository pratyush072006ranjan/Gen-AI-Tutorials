from dotenv import load_dotenv

load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
data=PyPDFLoader(r"C:\Users\ASUS\Documents\t\G_AI\Part-2\Document Loader\GRU.pdf")

docs=data.load()
print(len(docs))

