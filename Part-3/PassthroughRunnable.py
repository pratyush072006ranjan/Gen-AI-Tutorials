import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel

model=ChatMistralAI(model="mistral-small-2506")
parser=StrOutputParser()

code_prompt=ChatPromptTemplate.from_messages([
    ("system","You are a code generator"),
    ("human","{topic}")
])

explain_prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant who explains codes in simple terms"),
    ("human","Explain the following code in simple words:\n{code}")]
)

seq=code_prompt | model | parser

seq2=RunnableParallel(
    {
        "code":RunnablePassthrough(),
        "explanation": explain_prompt | model  | parser 
    }
)

chain= seq | seq2 
result=chain.invoke({"topic":"Write a code of palindrome in python"})

print(result["code"])
print("\n\n")
print(result["explanation"])
