from langchain_community.document_loaders import TextLoader

data=TextLoader(r"C:\Users\ASUS\Documents\t\G_AI\Part-2\Document Loader\notes.txt",
                    encoding="utf-8")
print(data)

docs=data.load()
print(docs[0])