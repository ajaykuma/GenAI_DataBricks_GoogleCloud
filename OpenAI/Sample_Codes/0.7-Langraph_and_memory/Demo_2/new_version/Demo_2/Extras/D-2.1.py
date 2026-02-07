# Integrating with memory safely in LangGraph + Streamlit

import os
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from openai import AzureOpenAI
from dotenv import load_dotenv

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

# -------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------
load_dotenv("E:\\Lesson_2_demos\\.env")

# Raw Azure OpenAI client (no memory)
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# -------------------------------------------------------------------
# Shared graph state
# -------------------------------------------------------------------
class State(TypedDict):
    product_name: str
    basic_description: str
    features_benefits: str
    marketing_message_premium: str
    marketing_message_loyalty: str
    marketing_message_corporate: str
    final_description: str

# -------------------------------------------------------------------
# Memory-safe LLM runner per session
# -------------------------------------------------------------------
SESSION_HISTORIES = {}
'''
This is a global dictionary, keyed by session_id.
Every time you call run_llm_with_session(prompt, session_id), it appends the prompt 
and response to SESSION_HISTORIES[session_id].
This is effectively your “memory buffer” for that session.
Refer: D-2.2.py for more details
'''

def run_llm_with_session(prompt: str, session_id: str = "product-session") -> str:
    """
    Run AzureChatOpenAI with memory tracked per session.
    """
    # Get or initialize session history
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = []

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("API_KEY"),
        api_version="2024-12-01-preview",
        deployment_name="gpt-4.1",
        temperature=1,
    )

    # Include prior messages for context
    prior_messages = SESSION_HISTORIES[session_id]
    response = llm(prior_messages + [HumanMessage(content=prompt)])

    # Save to session memory
    SESSION_HISTORIES[session_id].append(HumanMessage(content=prompt))
    SESSION_HISTORIES[session_id].append(response)

    return response.content

# -------------------------------------------------------------------
# Feature toggle
# -------------------------------------------------------------------
USE_MEMORY_FOR_FEATURES = True

# -------------------------------------------------------------------
# Graph nodes
# -------------------------------------------------------------------
def generate_basic_description(state: State) -> dict:
    output = run_llm_with_session(f"Write a brief description of a product named '{state['product_name']}'.")
    return {"basic_description": output}

def add_features_benefits_with_memory(state: State) -> dict:
    output = run_llm_with_session(f"List key features and benefits of the product: {state['basic_description']}")
    return {"features_benefits": output}

def add_features_benefits_without_memory(state: State) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": f"List key features and benefits of the product: {state['basic_description']}"}]
    )
    return {"features_benefits": response.choices[0].message.content}

def add_features_benefits(state: State) -> dict:
    if USE_MEMORY_FOR_FEATURES:
        return add_features_benefits_with_memory(state)
    else:
        return add_features_benefits_without_memory(state)

def create_marketing_message_premium(state: State) -> dict:
    output = run_llm_with_session(f"Create a premium marketing message for the product: {state['features_benefits']}")
    return {"marketing_message_premium": output}

def create_marketing_message_loyalty(state: State) -> dict:
    output = run_llm_with_session(f"Create a loyalty-based marketing message for the product: {state['features_benefits']}")
    return {"marketing_message_loyalty": output}

def create_marketing_message_corporate(state: State) -> dict:
    output = run_llm_with_session(f"Create a corporate marketing message for the product: {state['features_benefits']}")
    return {"marketing_message_corporate": output}

def polish_final_description(state: State) -> dict:
    prompt = (
        f"Combine and polish the product description including marketing messages:\n"
        f"Premium: {state['marketing_message_premium']}\n"
        f"Loyalty: {state['marketing_message_loyalty']}\n"
        f"Corporate: {state['marketing_message_corporate']}"
    )
    output = run_llm_with_session(prompt)
    return {"final_description": output}

# -------------------------------------------------------------------
# Build workflow
# -------------------------------------------------------------------
def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("generate_basic_description", generate_basic_description)
    workflow.add_node("add_features_benefits", add_features_benefits)
    workflow.add_node("create_marketing_message_premium", create_marketing_message_premium)
    workflow.add_node("create_marketing_message_loyalty", create_marketing_message_loyalty)
    workflow.add_node("create_marketing_message_corporate", create_marketing_message_corporate)
    workflow.add_node("polish_final_description", polish_final_description)

    workflow.add_edge(START, "generate_basic_description")
    workflow.add_edge("generate_basic_description", "add_features_benefits")

    workflow.add_edge("add_features_benefits", "create_marketing_message_premium")
    workflow.add_edge("add_features_benefits", "create_marketing_message_loyalty")
    workflow.add_edge("add_features_benefits", "create_marketing_message_corporate")

    workflow.add_edge("create_marketing_message_premium", "polish_final_description")
    workflow.add_edge("create_marketing_message_loyalty", "polish_final_description")
    workflow.add_edge("create_marketing_message_corporate", "polish_final_description")

    workflow.add_edge("polish_final_description", END)

    return workflow.compile()

# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------
def visualize_workflow():
    graph = nx.DiGraph()
    edges = [
        ("START", "generate_basic_description"),
        ("generate_basic_description", "add_features_benefits"),
        ("add_features_benefits", "create_marketing_message_premium"),
        ("add_features_benefits", "create_marketing_message_loyalty"),
        ("add_features_benefits", "create_marketing_message_corporate"),
        ("create_marketing_message_premium", "polish_final_description"),
        ("create_marketing_message_loyalty", "polish_final_description"),
        ("create_marketing_message_corporate", "polish_final_description"),
        ("polish_final_description", "END"),
    ]
    graph.add_edges_from(edges)
    plt.figure(figsize=(10, 6))
    nx.draw(
        graph,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=2000,
        font_size=10,
        font_weight="bold"
    )
    plt.savefig("workflow.png")

# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------
def run_streamlit_app():
    st.title("Product Description Generator with Memory Toggle")

    product_name = st.text_input("Enter the product name:")
    use_memory = st.checkbox("Use memory for Features & Benefits node", value=True)

    if st.button("Generate Product Description"):
        global USE_MEMORY_FOR_FEATURES
        USE_MEMORY_FOR_FEATURES = use_memory

        # Reset session memory per new run
        SESSION_HISTORIES.clear()

        initial_state = State(
            product_name=product_name,
            basic_description="",
            features_benefits="",
            marketing_message_premium="",
            marketing_message_loyalty="",
            marketing_message_corporate="",
            final_description=""
        )

        chain = build_workflow()
        result = chain.invoke(initial_state)

        st.subheader("Basic Description:")
        st.write(result["basic_description"])

        st.subheader("Features and Benefits:")
        st.write(result["features_benefits"])

        st.subheader("Premium Marketing Message:")
        st.write(result["marketing_message_premium"])

        st.subheader("Loyalty Marketing Message:")
        st.write(result["marketing_message_loyalty"])

        st.subheader("Corporate Marketing Message:")
        st.write(result["marketing_message_corporate"])

        st.subheader("Final Description:")
        st.write(result["final_description"])

        visualize_workflow()
        st.image("workflow.png", caption="Workflow")

# -------------------------------------------------------------------
if __name__ == "__main__":
    run_streamlit_app()
