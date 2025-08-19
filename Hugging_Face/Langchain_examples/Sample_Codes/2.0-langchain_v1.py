
#add your token here or refer:hf_api_1.py to use .env file to load token
api_token = ""

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint 

#Option 1
# repo_id = "google/flan-t5-base"  
# #alternatives:"flan-t5-large","flan-t5-xl" or "t5-large"

# llm = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     huggingfacehub_api_token=api_token,
#     temperature=0.5,
#     max_new_tokens=200
# )

# # New style: prompt | llm
# template = """Question: {question}
# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)

# # RunnableSequence
# chain = prompt | llm

# # Question 1
# question = "Explain the concept of black holes in simple terms."
# print(chain.invoke({"question": question}))

# # Question 2
# question = "What are the main causes of climate change, and how can we address them?"
# print(chain.invoke({"question": question}))

# # Question 3
# question = "Provide a brief overview of the history of artificial intelligence."
# print(chain.invoke({"question": question}))

#optimization
#Instead of repeating, we can loop
# questions = [
#     "Explain the concept of black holes in simple terms.",
#     "What are the main causes of climate change, and how can we address them?",
#     "Provide a brief overview of the history of artificial intelligence."
# ]

# for q in questions:
#     print(f"\nQ: {q}")
#     print(chain.invoke({"question": q}))

#Result
#The above code may fail as 
# StopIteration issue, Hugging Face’s InferenceClient can’t find a provider for the 
# model picked (google/flan-t5-base). 
# In plain terms → the model is on the Hub, but it isn’t served by the 
# nference API (no “warm” serverless endpoint).

# That’s why client.text_generation fails.

#Option 2
#Use a supported model
# repo_id = "HuggingFaceH4/zephyr-7b-alpha"

# llm = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     huggingfacehub_api_token=api_token,
#     temperature=0.5,
#     max_new_tokens=200
# )

# template = """Question: {question}
# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)

# chain = prompt | llm

# questions = [
#     "Explain the concept of black holes in simple terms.",
#     "What are the main causes of climate change, and how can we address them?",
#     "Provide a brief overview of the history of artificial intelligence."
# ]

# for q in questions:
#     print(f"\nQ: {q}")
#     print(chain.invoke({"question": q}))

#Result
#The above code may fail as ValueError: Model HuggingFaceH4/zephyr-7b-alpha is not supported for 
#task text-generation and provider featherless-ai. Supported task: conversational.

#Zephyr-7B is only exposed through Hugging Face Inference API under the conversational task (chat-like input/output), not text-generation.
#But HuggingFaceEndpoint defaults to the text-generation task.

#Option 3
#Using Inference client
from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-alpha", token=api_token)

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the concept of neaural networks in simple terms."}
    ],
    max_tokens=200,
    temperature=0.5
)

print(response.choices[0].message["content"])
