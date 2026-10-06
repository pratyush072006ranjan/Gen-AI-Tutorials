import warnings 
warnings.filterwarnings("ignore",category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


# Now create some tools

# weather tool

@tool
def get_weather(city:str)->str:
    """Get Current Weather of a city"""
    api_key=os.getenv("OPENWEATHER_API_KEY")
    url=f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response=requests.get(url)
    data=response.json()
    # print("DEBUG:",data)

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message','Could not fetch weather')}"

    temp=data["main"]["temp"]
    desc=data["weather"][0]["description"]

    return f"weather in {city}:{desc},{temp}°C"

# print(get_weather.invoke("Bilaspur"))

# Tavily news tool

tavily_client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city:str)->str:
    """Get Latest news about the city"""
    
    response=tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results=response.get("results",[])
    
    if not results:
        return f"No news found for {city}"
    
    news_list=[]

    for r in results:
        title=r.get("title","No Title")
        url=r.get("url","")
        snippet=r.get("content","")

        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )

    return f"latest news in {city}:\n\n" + "\n\n".join(news_list)

# print(get_news.invoke("Bilaspur"))

llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

@wrap_tool_call
def human_approval(request,handler):
    """Ask for human approval before every tool call."""
    tool_name=request.tool_call["name"]
    confirm=input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower()!="yes":
        return ToolMessage(
            content="Tool Call denied by user.",
            tool_call_id=request.tool_call["id"]
        )
    return handler(request)

agent=create_agent(
    llm,
    tools=[get_weather,get_news],
    system_prompt="you are a helpful city assitant.",
    middleware=[human_approval]
)

print("City Agent")
print("Type exit to quit")

while True:
    user_input=input("You: ")

    if user_input.lower()=="exit":
        break

    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )

    # print("bot: ",result['messages'][-1].content)
    print(result)