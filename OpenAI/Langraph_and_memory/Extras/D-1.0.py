#Using a chat model instead of completion model
#AzureOpenAI class from langchain_openai, which by default creates a completion model,
# not a chat model.
import os
import streamlit as st
from pydantic import BaseModel
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize Azure OpenAI client
client = AzureChatOpenAI(
section...
)

# Define a structured schema (example)
class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str

# Streamlit UI
st.title("Web Search Optimization with LLM")
st.write("Enter a question to receive an optimized web search query and reasoning.")

user_query = st.text_input("Enter your question:")

if user_query:
    response = client.invoke([
        HumanMessage(content=f"Generate an optimized search query and explain why for: '{user_query}'")
    ])
    
    st.subheader("Model Response:")
    st.write(response.content)

    # Optional: parse response
    try:
        parts = response.content.split("\n\n", 1)
        formatted_response = WebSearchPrompt(
            search_query=parts[0].strip(),
            justification=parts[1].strip() if len(parts) > 1 else "No justification provided."
        )

        st.subheader("Optimized Search Query:")
        st.write(formatted_response.search_query)

        st.subheader("Reasoning:")
        st.write(formatted_response.justification)
    except Exception as e:
        st.error(f"Failed to parse response: {e}")
