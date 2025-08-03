import os
import streamlit as st
from pydantic import BaseModel, ValidationError
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import json

# Initialize Azure OpenAI client
client = AzureChatOpenAI(
section...
)

# Define structured response schema
class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str

# Streamlit UI
st.title("Web Search Optimization with LLM")
user_query = st.text_input("Enter your question:")

if user_query:
    # Prompt the model to return clean JSON
    prompt = f"""
    Based on the user's question, generate an optimized web search query and provide reasoning.

    Respond ONLY with raw JSON (no markdown or code blocks). Format:
    {{
        "search_query": "...",
        "justification": "..."
    }}

    User question: "{user_query}"
    """

    # Get model response
    response = client.invoke([HumanMessage(content=prompt)])
    raw_response = response.content

    st.subheader("Raw Model Response:")
    st.code(raw_response, language="json")

    # Clean and parse response
    try:
        cleaned_json = raw_response.strip()
        if cleaned_json.startswith("```json") or cleaned_json.startswith("```"):
            cleaned_json = cleaned_json.strip("`").split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        
        parsed = json.loads(cleaned_json)
        structured = WebSearchPrompt(**parsed)

        st.subheader("Optimized Search Query:")
        st.write(structured.search_query)

        st.subheader("Reasoning:")
        st.write(structured.justification)

    except (json.JSONDecodeError, ValidationError) as e:
        st.error(f"Failed to parse structured JSON: {e}")
