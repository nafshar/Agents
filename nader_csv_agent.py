
# ! pip install cohere langchain-cohere langchain-experimental -qq

#import json
import os
import pandas as pd
# import cohere
# import langchain
# import langchain_experimental
from langchain_cohere import ChatCohere, create_csv_agent
from dotenv import load_dotenv
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
os.environ['COHERE_API_KEY'] = COHERE_API_KEY

df = pd.read_csv('evaluation_results.csv')
df.head()

# Define the Cohere LLM
llm = ChatCohere(cohere_api_key=COHERE_API_KEY,
                 model="command-r-plus-08-2024",
                 temperature=0)
agent_executor = create_csv_agent(
    llm,
    "evaluation_results.csv"
)
resp = agent_executor.invoke({"input":"What's the average evaluation score in run A?"})
print(resp.get("output"))

resp = agent_executor.invoke({"input":"What's the latency of the highest-scoring run for the summarize_article use case?"})
print(resp.get("output"))

resp = agent_executor.invoke({"input":"which usecase has the lowest latency and what is that value?"})
print(resp.get("output"))

resp = agent_executor.invoke({"input":"which usecase named extract_names has the lowest latency and what is that value?"})
print(resp.get("output"))

resp = agent_executor.invoke({"input":"for each usecase show the lowest value?"})
print(resp.get("output"))

resp = agent_executor.invoke({"input":"show a chart of latency values?"})
print(resp.get("output"))