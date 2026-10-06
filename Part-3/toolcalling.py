import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from langchain_mistralai import ChatMistralAI
from langchain.tools import tools
from rich import print

# 1. creating a tool
@tool
def get_text_length(text:str)->int:
    """Returns the number of character in a given text"""
    return len(text)

llm=ChatMistralAI(model="mistral-small-2506")

# tool binding
llm_with_tool=llm.bind_tools([get_text_length])

# result=llm.invoke("Hello")

# result2=llm_with_tool.invoke("Hello")

result=llm.invoke("Returns the number of character in a given text : 'hello how are you' ")

result2=llm_with_tool.invoke("Returns the number of character in a given text : 'hello how are you' ")

print(result)
print()
print()
print()
print(result2)
