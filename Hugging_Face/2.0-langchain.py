# loading models from Hugging Face Hub
# #pip install langchain langchain-community langchain-core

from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

#add your token here or refer:hf_api_1.py to use .env file to load token
api_token = "-----"

# repo_id = "google/flan-t5-large"  

# #alternatives:"flan-t5-large","flan-t5-xl" or "t5-large"
# llm=HuggingFaceHub(repo_id=repo_id, 
#                    model_kwargs={"temperature":1, "max_length":512},
#                    task="text-generation",
#                    huggingfacehub_api_token=api_token)

# template = """Question: {question}
# Answer: Let's think step by step."""
# prompt = PromptTemplate(template=template, input_variables=["question"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# #Question 1
# question = "Explain the concept of black holes in simple terms."
# llm_chain.run(question)

# #Question 2
# question = "What are the main causes of climate change, and how can we address them?"
# llm_chain.run(question)

# #Question 3
# question = "Provide a brief overview of the history of artificial intelligence."
# llm_chain.run(question)

#Above code shows error as model too large or other warnings of deprecation
#such as use task methods instead (e.g. InferenceClient.chat_completion)

from huggingface_hub import InferenceClient

# Initialize the InferenceClient
client = InferenceClient(model="google/flan-t5-large", token=api_token)

# Your prompt
question = "Explain the concept of black holes in simple terms."
prompt = f"Question: {question}\nAnswer:"

# Call the model (this uses text-to-text generation under the hood)
response = client.text_generation(prompt, max_new_tokens=100)

print(response)

#might throw warnings: Importing HuggingFaceHub from langchain root module is no longer supported. 
#Please use langchain_community.llms.HuggingFaceHub instead
#and also shows response
#response
    #Black holes are the result of a black hole collapsing into itself.

