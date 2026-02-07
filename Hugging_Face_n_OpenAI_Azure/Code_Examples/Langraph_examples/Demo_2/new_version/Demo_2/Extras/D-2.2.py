import os
import copy
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from openai import AzureOpenAI
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv("E:\\Lesson_2_demos\\.env")

chat_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1",
    temperature=0.7,
)

raw_client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# --------------------------------------------------
# LangGraph state (NOT memory)
# --------------------------------------------------
class State(TypedDict):
    product_name: str
    basic_description: str
    features_benefits: str
    marketing_message_premium: str
    marketing_message_loyalty: str
    marketing_message_corporate: str
    final_description: str

# --------------------------------------------------
# Graph nodes
# --------------------------------------------------
def generate_basic_description(state: State):
    response: AIMessage = chat_llm.invoke([
        HumanMessage(
            content=f"Write a brief description of a product named '{state['product_name']}'."
        )
    ])
    return {"basic_description": response.content}

def add_features_benefits(state: State):
    response = raw_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": f"List key features and benefits of: {state['basic_description']}"
        }]
    )
    return {"features_benefits": response.choices[0].message.content}

def create_marketing_message_premium(state: State):
    response: AIMessage = chat_llm.invoke([
        HumanMessage(
            content=f"Create a premium marketing message for: {state['features_benefits']}"
        )
    ])
    return {"marketing_message_premium": response.content}

def create_marketing_message_loyalty(state: State):
    response: AIMessage = chat_llm.invoke([
        HumanMessage(
            content=f"Create a loyalty marketing message for: {state['features_benefits']}"
        )
    ])
    return {"marketing_message_loyalty": response.content}

def create_marketing_message_corporate(state: State):
    response: AIMessage = chat_llm.invoke([
        HumanMessage(
            content=f"Create a corporate marketing message for: {state['features_benefits']}"
        )
    ])
    return {"marketing_message_corporate": response.content}

def polish_final_description(state: State):
    response: AIMessage = chat_llm.invoke([
        HumanMessage(
            content=(
                f"Combine and polish:\n"
                f"Premium: {state['marketing_message_premium']}\n"
                f"Loyalty: {state['marketing_message_loyalty']}\n"
                f"Corporate: {state['marketing_message_corporate']}"
            )
        )
    ])
    return {"final_description": response.content}

# --------------------------------------------------
# Build graph
# --------------------------------------------------
def build_workflow():
    graph = StateGraph(State)

    graph.add_node("basic", generate_basic_description)
    graph.add_node("features", add_features_benefits)
    graph.add_node("premium", create_marketing_message_premium)
    graph.add_node("loyalty", create_marketing_message_loyalty)
    graph.add_node("corporate", create_marketing_message_corporate)
    graph.add_node("final", polish_final_description)

    graph.add_edge(START, "basic")
    graph.add_edge("basic", "features")

    graph.add_edge("features", "premium")
    graph.add_edge("features", "loyalty")
    graph.add_edge("features", "corporate")

    graph.add_edge("premium", "final")
    graph.add_edge("loyalty", "final")
    graph.add_edge("corporate", "final")

    graph.add_edge("final", END)

    return graph.compile()

# --------------------------------------------------
# Visualization
# --------------------------------------------------
def visualize_workflow():
    g = nx.DiGraph()
    edges = [
        ("START", "basic"),
        ("basic", "features"),
        ("features", "premium"),
        ("features", "loyalty"),
        ("features", "corporate"),
        ("premium", "final"),
        ("loyalty", "final"),
        ("corporate", "final"),
        ("final", "END"),
    ]
    g.add_edges_from(edges)
    plt.figure(figsize=(10, 5))
    nx.draw(g, with_labels=True, node_color="lightblue", node_size=2000)
    plt.savefig("workflow.png")

# --------------------------------------------------
# Streamlit app (memory owner)
# --------------------------------------------------
def run_streamlit_app():
    st.title("Product Description Generator (Graph + Cache)")

    if "PRODUCT_MEMORY" not in st.session_state:
        st.session_state.PRODUCT_MEMORY = {}

    product_name = st.text_input("Enter product name")

    if st.button("Generate") and product_name:
        product_name = product_name.strip().lower()

        # MEMORY CHECK
        if product_name in st.session_state.PRODUCT_MEMORY:
            st.info("Loaded from memory — no LLM call")
            result = st.session_state.PRODUCT_MEMORY[product_name]
        else:
            st.info("New product — running LangGraph")
            chain = build_workflow()

            initial_state = State(
                product_name=product_name,
                basic_description="",
                features_benefits="",
                marketing_message_premium="",
                marketing_message_loyalty="",
                marketing_message_corporate="",
                final_description=""
            )

            result = chain.invoke(initial_state)
            st.session_state.PRODUCT_MEMORY[product_name] = copy.deepcopy(result)

        # Debug memory snapshot
        st.write("MEMORY SNAPSHOT:", {
            k: "cached" for k in st.session_state.PRODUCT_MEMORY
        })

        # Display
        st.subheader("Basic Description")
        st.write(result["basic_description"])

        st.subheader("Features & Benefits")
        st.write(result["features_benefits"])

        st.subheader("Premium Marketing")
        st.write(result["marketing_message_premium"])

        st.subheader("Loyalty Marketing")
        st.write(result["marketing_message_loyalty"])

        st.subheader("Corporate Marketing")
        st.write(result["marketing_message_corporate"])

        st.subheader("Final Description")
        st.write(result["final_description"])

        visualize_workflow()
        st.image("workflow.png", caption="Workflow")

# --------------------------------------------------
if __name__ == "__main__":
    run_streamlit_app()
