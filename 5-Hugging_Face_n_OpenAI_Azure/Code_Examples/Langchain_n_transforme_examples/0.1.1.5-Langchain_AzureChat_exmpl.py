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
    api_version=api_version,
    deployment_name=deployment,  # e.g. "gpt-4o-mini"
)

response = llm.invoke("Hello! How are you?")
print(response)
