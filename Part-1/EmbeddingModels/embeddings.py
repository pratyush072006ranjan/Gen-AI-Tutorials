from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=64
)

texts=[
    "Hello this is Akarsh Vyas",
    "Hello your name is YouTube",
    "And you all are very beautiful"
]
# vector = embeddings.embed_query(
#     "You are going to learn Gen AI"
# )
vector = embeddings.embed_documents(
    "You are going to learn Gen AI"
)

print(vector)
print("Dimensions:", len(vector))

# result = client.models.embed_content(
#     model="gemini-embedding-001",
#     contents="This is the text I want to embed."
# )

# embedding = result.embeddings[0].values

# print(len(embedding))
# print(embedding[:10])