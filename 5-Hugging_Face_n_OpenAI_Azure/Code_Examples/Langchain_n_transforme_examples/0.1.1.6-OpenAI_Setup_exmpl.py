#!pip install dotenv
#!pip install openai

#Working with newer version of GPT & OpenAI directly

import os
import openai
from openai import OpenAI

openai_base_url = os.getenv("OPENAI_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-3.5-turbo"

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

response = client.responses.create(
    model=model,
    input="What is the capital of France?",
)

print(f"answer: {response.output[0]}")
