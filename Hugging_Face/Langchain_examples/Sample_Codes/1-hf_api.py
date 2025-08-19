#Remarks:
# Not every model on the Hub qualifies—even some popular models are unsupported or require Pro. 
# To find supported ones, search on the Models page and filter by “inference provider” availability, 
# or query programmatically via the Hub API.
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint 

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#repo_id = "tiiuae/falcon-7b-instruct"
repo_id = "mixtral-8x7b-instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=hf_token,
    temperature=0.5,
    max_new_tokens=200
)

# Use .invoke instead of calling directly
response = llm.invoke("What was the first Disney movie?")
print(response)
