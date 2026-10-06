import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

model=ChatMistralAI(model="mistral-small-2506")
parser=StrOutputParser()

# Two different prompts
short_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)


# Manual Method
# # Input
# topic="Machine Learning"

# formatted_short=short_prompt.format_messages(
#     topic=topic
# )

# response_short=model.invoke(formatted_short)

# str_out=parser.parse(response_short.content)


# formatted_long=detailed_prompt.format_messages(
#     topic=topic
# )

# response_long=model.invoke(formatted_long)

# str_out_long=parser.parse(response_long.content)

# Using Parallel Runnables
# topic="Machine Learning"

chain=RunnableParallel({
"short" :RunnableLambda(lambda x:x["short"]) | short_prompt | model | parser ,
"detailed" :RunnableLambda(lambda x:x["detailed"]) | detailed_prompt | model | parser
})


# If we want summary of different topics from  them

result=chain.invoke({
    "short": {"topic" : "Machine Learning"},
    "detailed": {"topic":"Deep Learning"}
    })

print("\n======Short Explanation========\n")
print(result['short'])


print("\n======Detailed Explanation========\n")
print(result['detailed'])
