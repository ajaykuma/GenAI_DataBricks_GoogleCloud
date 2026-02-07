# Step 0: Import required libraries

#from dotenv import load_dotenv
import os
import streamlit as st
from pydantic import BaseModel
import openai

from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv("E:\\Lesson_2_demos\\.env")

client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

#If using AzureChatOpenAI
# from langchain_openai import AzureChatOpenAI

# from dotenv import load_dotenv
# load_dotenv("E:\\Lesson_2_demos\\.env")

# ## Start by creating an instance of the AzureChatOpenAI class.
# client = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("API_KEY"),
#     api_version="2024-12-01-preview",
#     deployment_name="gpt-5.1-chat",
#     temperature=0,
# )

# Step 2: Define a structured schema using Pydantic
# This schema will ensure that the LLM output is structured and validated.
class WebSearchPrompt(BaseModel):
    search_query: str
    llm_response: str

# Step 3: Build Streamlit UI
# Use Streamlit to create a simple UI where users can input their question and get a structured response.
st.title("Web Search Optimization with LLM")
st.write("Enter a question to receive an optimized web search query and reasoning.")

# Step 4: Create input field for the user's question
user_query = st.text_input("Enter your question:") # ask question

# Step 5: Process the input query and display the result
if user_query:
    # Invoke the LLM with the user query
    # Extract relevant parts of the response
    response = client.chat.completions.create(model="gpt-4.1",
                                          messages=[{"role": "user", "content": user_query}])
                                          #temperature=0.3)
    response_content = response.choices[0].message.content

    # Structure the output using the pydantic model
    formatted_response = WebSearchPrompt(search_query=user_query, llm_response=response_content)


    # Display the structured response to the user
    st.subheader("Optimized Search Query:")
    st.write(formatted_response.search_query)  # Display the optimized search query
    st.subheader("Response:")
    st.write(formatted_response.llm_response)   # Display the reasoning behind the query
