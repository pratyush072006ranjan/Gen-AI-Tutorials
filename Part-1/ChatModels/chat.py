from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "google_genai:gemini-flash-latest"
)

model2 = init_chat_model(
    "mistralai:mistral-small-2506",
    temperature=0.9,
    max_tokens=20
)
model3 = init_chat_model(
    "groq:llama-3.1-8b-instant"
)
response = model3.invoke("what is cricket?")
print(response.text)