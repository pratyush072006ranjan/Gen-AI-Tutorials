import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from rich import print
from langchain_core.messages import HumanMessage


# 1. creating a tool
@tool
def get_text_length(text:str)->int:
    """Returns the number of character in a given text"""
    return len(text)

llm=ChatMistralAI(model="mistral-small-2506")

# tool binding
llm_with_tool=llm.bind_tools([get_text_length])

# result=llm_with_tool.invoke("Returns the number of character in a given text : 'hello how are you' ")

# if result.tool_calls:
#     tool_call=result.tool_calls[0]

# tool_name=tool_call["name"]
# tool_args=tool_call["args"]

# tool_result=get_text_length.invoke(total_args)

# final_response=llm_with_tool.invoke(f"The length of text is {tool_result}")

# print(final_response.content)

tools={
    "get_text_length":get_text_length
}
message=[]
prompt=input("You: ")
query=HumanMessage(content=prompt)
message.append(query)

result=llm_with_tool.invoke(message)


message.append(result)


if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

    result = llm_with_tool.invoke(message)
    print(result.content)
else:
    print(result.content)