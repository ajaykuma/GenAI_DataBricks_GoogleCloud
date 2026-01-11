import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

api_key = os.getenv("API_KEY")

from langchain_openai import AzureOpenAI

llm = AzureOpenAI(
    azure_endpoint="https://myaiwork.cognitiveservices.azure.com/",
    api_key=api_key,
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1",  # same deployment name
)

response = llm.invoke("Hello! How are you?")
print(response)

#Might throw
#'The completion operation does not work with the specified model, gpt-4.1.
