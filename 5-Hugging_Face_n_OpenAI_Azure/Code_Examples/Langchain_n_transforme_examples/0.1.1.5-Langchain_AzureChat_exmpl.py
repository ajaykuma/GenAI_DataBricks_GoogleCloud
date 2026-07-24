import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

api_key = os.getenv("API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_API_VERSION")
deployment = os.getenv("AZURE_DEPLOYMENT_NAME")

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version="2025-04-01-preview",
    deployment_name="gpt-4.1"
)

#QnA
response = llm.invoke("well this time, i am going to budapest, suggest what can i see and what can i eat")
print(response)

#Generating content
# response = llm.invoke("Generate a marketing campaign for my new product - non-plastic water bottle. " \
# "the targeted audience would be between age group of 15-28 and mainly travellers")
# print(response)

#Code generation
# response = llm.invoke("Generate a simple code using python which can build a QA Agent that takes" \
# "inputs from developers ad generates test cases." \
# "Make sure that the example uses gpt-5.1 and can use AzureChatOpenAI to work with llm")

# print(type(response))

# print(response.content)

#Analysis
# mydata = "https://raw.githubusercontent.com/ajaykuma/Datasets_For_Work/refs/heads/main/Bank_full.csv"
# response = llm.invoke(f"Analyze the data provided in {mydata} and give me some insights based on data")

# print(response.content)


