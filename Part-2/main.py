# from dotenv import load_dotenv

# load_dotenv()

# from langchain_mistralai import ChatMistralAI
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # data=TextLoader(r"C:\Users\ASUS\Documents\t\G_AI\Part-2\Document Loader\notes.txt",encoding="utf-8")
# data=PyPDFLoader(r"C:\Users\ASUS\Documents\t\G_AI\Part-2\Document Loader\deeplearningG.pdf")
# docs=data.load()
# template=ChatPromptTemplate.from_messages(
#     [("system","you are a AI that summarizes the text"),("human","{data}")]
# )
# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks=splitter.split_documents(docs)

# model=ChatMistralAI(model="mistral-small-2506")

# prompt=template.format_messages(data=docs[0].page_content)
# result=model.invoke(prompt)

# print(result.content)

import warnings 
warnings.filterwarnings("ignore",category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

embedding_model=MistralAIEmbeddings()

vectorstore=Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

# prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",
        """You are a helpful AI Assistant.
        Use ONLY the provided context to answer the question.

        If the answer is not present in the context,
        say:"I could not find the answer in the document."
        """
        ),
        (
            "human",
            """Context:
            {context}
            
            Question:
            {question}
            """
        )
    ]
)

print("RAG System Created...")

print("press 0 to exit")

while True:
    query=input("You:")
    if query=="0":
        break

    docs=retriever.invoke(query)

    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })


    response=llm.invoke(final_prompt)

    print(f"AI: {response.content[0]['text']}")
    print()
