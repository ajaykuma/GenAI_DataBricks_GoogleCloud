import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

api_key = os.getenv("API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1",  # e.g. "gpt-4o-mini"
)

response = llm.invoke("Hello! How are you?")
print(response)
