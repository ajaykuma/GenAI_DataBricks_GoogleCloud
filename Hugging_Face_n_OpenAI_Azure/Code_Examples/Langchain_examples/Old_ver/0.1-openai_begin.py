#pip install --upgrade openai 
import os
import openai
from openai import OpenAI
client = OpenAI(api_key="")

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file
# #load_dotenv()
# OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
# client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model="gpt-4o-mini",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "user", "content": "Tell a good joke in the form of a question. Do not yet give the answer."}
  ]
)
print(completion.choices[0].message.content)

