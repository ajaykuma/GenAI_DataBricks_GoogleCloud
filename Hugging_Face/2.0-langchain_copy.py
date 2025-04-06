# loading models from Hugging Face Hub
# #pip install langchain langchain-community langchain-core

#from langchain import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

#add your token here or refer:hf_api_1.py to use .env file to load token
api_token = ""

from huggingface_hub import InferenceClient

# Initialize the InferenceClient
client = InferenceClient(model="google/flan-t5-large", token=api_token)

# Your prompt
# question = "Explain the concept of black holes in simple terms."
# prompt = f"Question: {question}\nAnswer:"

# # Call the model (this uses text-to-text generation under the hood)
# response = client.text_generation(prompt, max_new_tokens=100)

# print(response)

#might throw warnings: Importing HuggingFaceHub from langchain root module is no longer supported. 
#Please use langchain_community.llms.HuggingFaceHub instead
#and also shows response
#response
    #Black holes are the result of a black hole collapsing into itself.

question = """
Let's consider which is heavier: 1000 feathers or a 30-pound weight. I'll think through this in a few different ways and then decide which answer seems most consistent.

1. First line of reasoning: A single feather is very light, almost weightless. So, 1000 feathers might still be quite light, possibly lighter than a 30-pound weight.

2. Second line of reasoning: 1000 is a large number, and when you add up the weight of so many feathers, it could be quite heavy. Maybe it's heavier than a 30-pound weight.

3. Third line of reasoning: The average weight of a feather is very small. Even 1000 feathers would not add up to 30 pounds.

Considering these reasonings, the most consistent answer is:
"""

prompt = f"Question: {question}\nAnswer:"

response = client.text_generation(prompt)

print(response)
