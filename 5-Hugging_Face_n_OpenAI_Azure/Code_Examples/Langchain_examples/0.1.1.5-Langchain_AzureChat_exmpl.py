import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    deployment_name="gpt-4.1",  # same deployment name
)


response = llm.invoke("Hello! How are you?")
print(response)

#Might throw
#'The completion operation does not work with the specified model, gpt-4.1.
