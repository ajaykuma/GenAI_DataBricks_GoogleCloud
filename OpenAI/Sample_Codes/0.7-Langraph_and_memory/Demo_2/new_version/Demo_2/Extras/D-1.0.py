#Using a chat model instead of completion model
#AzureOpenAI class from langchain_openai, which by default creates a completion model,
# not a chat model.
import os
import streamlit as st
import openai
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv("E:\\Lesson_2_demos\\.env")

#If using AzureOpenAI
#from openai import AzureOpenAI
# client = AzureOpenAI(
#     api_key=os.getenv("API_KEY"),
#     api_version="2024-12-01-preview",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# )

# user_query = st.text_input("Enter your question:")

# response = client.chat.completions.create(
#     model="gpt-4.1",  # your deployment name
#     messages=[
#         {
#             "role": "user",
#             "content": f"Generate an optimized search query and explain why for: '{user_query}'"
#         }
#     ],
# )

# st.write(response.choices[0].message.content)

#If using AzureChatOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# ## Start by creating an instance of the AzureChatOpenAI class.
client = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1",
    temperature=0,
)

#Define a structured schema (example)
class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str

# Streamlit UI
st.title("Web Search Optimization with LLM")
st.write("Enter a question to receive an optimized web search query and reasoning.")

user_query = st.text_input("Enter your question:")

if user_query:
    response = client.invoke(
       [HumanMessage(content=f"Generate an optimized search query and explain \
                     why for the: '{user_query}' and then generate the response \
                     to the optimized query")])
    
    #Uncomment below lines to see raw response
    #st.subheader("Model Response:")
    #st.write(response.content)

    #Comment out earlier message and Try this prompt
    # [HumanMessage(content=f""" 
    #     Generate:
    #     1. ONE optimized web search query (write it once, no headings)
    #     2. ONE short explanation for why this query is effective
    #     3. ONE response to the optimized query

    #     Rules:
    #     - Do NOT repeat the search query
    #     - Do NOT repeat section titles
    #     - Use plain text, clearly separated by new lines

    #     User question: "{user_query}" """)])
    
    # Optional: parse response
    try:
        parts = response.content.split("\n\n", 1)
        formatted_response = WebSearchPrompt(
            search_query=parts[0].strip(),
            justification=parts[1].strip() if len(parts) > 1 else "No justification provided."
        )

        st.subheader("Optimized Search Query:")
        st.write(formatted_response.search_query)

        st.subheader("Reasoning & Response:")
        st.write(formatted_response.justification)
    except Exception as e:
        st.error(f"Failed to parse response: {e}")
