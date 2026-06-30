#!pip install dotenv
#!pip install openai
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load variables from .env (Located in project folder here)
load_dotenv()

api_key = os.getenv("API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_DEPLOYMENT_NAME")
deployment = model_name
api_version = os.getenv("AZURE_API_VERSION")

#Working with GPT-4.1
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see & what should i eat?",
        }
    ],
    max_completion_tokens=13107,
    temperature=1.0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    model=deployment
)

print(response.choices[0].message.content)

#Working with newer version of GPT (5.1) & OpenAI
#deploy a base model: gpt-5.1
#update .env file to point to your endpoints,api-key and api-version
api_key = os.getenv("API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_DEPLOYMENT_NAME")
deployment = model_name
api_version = os.getenv("AZURE_API_VERSION")
# from openai import OpenAI

# endpoint = endpoint
# deployment_name = deployment
# api_key = api_key

# client = OpenAI(
#     base_url=endpoint,
#     api_key=api_key
# )

# response = client.responses.create(
#     model=deployment_name,
#     input="What is the capital of France?",
# )

# print(f"answer: {response.output[0]}")
