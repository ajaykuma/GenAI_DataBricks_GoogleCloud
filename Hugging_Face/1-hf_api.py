#pip install langchain langchain-community langchain-core
#pip install huggingface_hub
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"]

from langchain import HuggingFaceHub
repo_id = "tiiuae/falcon-7b-instruct"  
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 1000}
)
print(llm("what was the first disney movie?"))

