from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

model=ChatMistralAI(model='mistral-small-2506',temperature=0.9)

print("Choose your AI Mode:")
print("press 1 for Angry Mode.")
print("press 2 for Funny Mode")
print("press 3 for Sad Mode")
choice=int(input("Tell your response:-"))

if choice==1:
    mode="You are an angry AI agent.You respond aggrasively and impatiently."
elif choice==2:
    mode="You are a funny AI Agent.You respond with humor and jokes."
elif choice==3:
    mode="You are a sad AI agent.You respond with depressing words."

messages=[
    SystemMessage(content=mode)
]

print("------------------Welcome--------------------------------")
print("Type 0 to exit")
while True:
    
    prompt=input("You :")
    messages.append(HumanMessage(content=prompt))
    if prompt=="0":
        break
    response=model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot:" ,response.content)


print(messages)
