from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# Load environment variables (optional)
load_dotenv()

# Initialize Ollama model (local)
llm = Ollama(model="llama3")  # Make sure llama3 is running: `ollama run llama3`

# --- State Type ---
class State(TypedDict):
    product_name: str
    basic_description: str
    features_benefits: str
    marketing_message: str
    final_description: str

# --- Graph Node Functions ---

def generate_basic_description(state):
    prompt = f"Write a brief description of a product named '{state['product_name']}'."
    return {"basic_description": llm.invoke(prompt)}

def add_features_benefits(state: State):
    prompt = f"List key features and benefits of the product: {state['basic_description']}"
    return {"features_benefits": llm.invoke(prompt)}

def create_marketing_message(state: State):
    prompt = f"Create a compelling marketing message for the product based on: {state['features_benefits']}"
    return {"marketing_message": llm.invoke(prompt)}

def polish_final_description(state: State):
    prompt = (
        f"Polish and finalize the product description based on the following details:\n"
        f"Product Name: {state['product_name']}\n"
        f"Basic Description: {state['basic_description']}\n"
        f"Features & Benefits: {state['features_benefits']}\n"
        f"Marketing Message: {state['marketing_message']}"
    )
    return {"final_description": llm.invoke(prompt)}

# --- Workflow Builder ---

def build_workflow():
    workflow = StateGraph(State)
    workflow.add_node("generate_basic_description", generate_basic_description)
    workflow.add_node("add_features_benefits", add_features_benefits)
    workflow.add_node("create_marketing_message", create_marketing_message)
    workflow.add_node("polish_final_description", polish_final_description)

    workflow.add_edge(START, "generate_basic_description")
    workflow.add_edge("generate_basic_description", "add_features_benefits")
    workflow.add_edge("add_features_benefits", "create_marketing_message")
    workflow.add_edge("create_marketing_message", "polish_final_description")
    workflow.add_edge("polish_final_description", END)

    return workflow.compile()

# --- Visualize Workflow ---

def visualize_workflow():
    graph = nx.DiGraph()
    edges = [
        ("START", "generate_basic_description"),
        ("generate_basic_description", "add_features_benefits"),
        ("add_features_benefits", "create_marketing_message"),
        ("create_marketing_message", "polish_final_description")
    ]
    graph.add_edges_from(edges)

    plt.figure(figsize=(8, 5))
    nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray',
            node_size=2000, font_size=10, font_weight='bold')
    plt.savefig("workflow.png")

# --- Streamlit UI ---

def run_streamlit_app():
    st.title("🛠️ Product Description Generator (Ollama)")
    product_name = st.text_input("Enter the product name:")

    if st.button("Generate Product Description") and product_name:
        state = State(
            product_name=product_name,
            basic_description="",
            features_benefits="",
            marketing_message="",
            final_description=""
        )

        chain = build_workflow()
        result = chain.invoke(state)

        st.subheader("📄 Basic Description:")
        st.write(result["basic_description"])

        st.subheader("✨ Features and Benefits:")
        st.write(result["features_benefits"])

        st.subheader("📣 Marketing Message:")
        st.write(result["marketing_message"])

        st.subheader("✅ Final Description:")
        st.write(result["final_description"])

        visualize_workflow()
        st.image("workflow.png", caption="Product Description Workflow")

if __name__ == "__main__":
    run_streamlit_app()
