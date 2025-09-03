# Step 0: Import required libraries

#from dotenv import load_dotenv
import os
import streamlit as st
from pydantic import BaseModel
import openai

# Step 1: Create a client 
client = openai.AzureOpenAI(
    section...
)

# Step 2: Define a structured schema using Pydantic
# This schema will ensure that the LLM output is structured and validated.
class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str

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
    response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[{"role": "user", "content": user_query}],
                                          temperature=0.3)
    response_content = response.choices[0].message.content

    # Structure the output using the pydantic model
    formatted_response = WebSearchPrompt(search_query=user_query, justification=response_content)


    # Display the structured response to the user
    st.subheader("Optimized Search Query:")
    st.write(formatted_response.search_query)  # Display the optimized search query
    st.subheader("Reasoning:")
    st.write(formatted_response.justification)   # Display the reasoning behind the query
