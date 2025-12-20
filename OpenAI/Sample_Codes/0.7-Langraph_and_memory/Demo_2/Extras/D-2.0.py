import openai
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from dataclasses import asdict, dataclass
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# OpenAI API setup (ensure you have your OpenAI API key set)
#load_dotenv()
#api_key = os.getenv("OPENAI_API_KEY")
#client = openai.OpenAI()

client = AzureChatOpenAI(
section...
)

# --- Define the shared state ---
class State(TypedDict):
    product_name: str
    basic_description: str
    features_benefits: str
    marketing_message_premium: str
    marketing_message_loyalty: str
    marketing_message_corporate: str
    final_description: str

# --- Workflow nodes --- (Step 3-6 extended)
def generate_basic_description(state: State):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates brief product descriptions."},
            {"role": "user", "content": f"Write a brief description of a product named '{state['product_name']}'."}
        ]
    )
    return {"basic_description": response.choices[0].message.content}

def add_features_benefits(state: State):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"List key features and benefits of the product: {state['basic_description']}"}
        ]
    )
    return {"features_benefits": response.choices[0].message.content}

def create_marketing_message_premium(state: State):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Create a premium/luxury-style marketing message for: {state['features_benefits']}"}
        ]
    )
    return {"marketing_message_premium": response.choices[0].message.content}

def create_marketing_message_loyalty(state: State):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Create a personalized loyalty-style message for returning customers based on: {state['features_benefits']}"}
        ]
    )
    return {"marketing_message_loyalty": response.choices[0].message.content}

def create_marketing_message_corporate(state: State):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Create a corporate-style marketing message for business clients based on: {state['features_benefits']}"}
        ]
    )
    return {"marketing_message_corporate": response.choices[0].message.content}

def polish_final_description(state: State):
    full_content = f"Premium: {state['marketing_message_premium']}\n\nLoyalty: {state['marketing_message_loyalty']}\n\nCorporate: {state['marketing_message_corporate']}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Combine and polish the following marketing messages into a cohesive final description for all audiences:\n{full_content}"}
        ]
    )
    return {"final_description": response.choices[0].message.content}
# Step 7: Function to build the workflow (Separate from Streamlit logic)

# def build_workflow():
#     """Build and compile the workflow steps using LangGraph."""
#     # Build the workflow for generating the product description
#     workflow = StateGraph(State)
    
#     # Add nodes to the workflow (steps in the process)
#     workflow.add_node("generate_basic_description", generate_basic_description)  # Step 1: Basic Description
#     workflow.add_node("add_features_benefits", add_features_benefits)  # Step 2: Features and Benefits
#     workflow.add_node("create_marketing_message", create_marketing_message)  # Step 3: Marketing Message
#     workflow.add_node("polish_final_description", polish_final_description)  # Step 4: Final Description

#     # Add edges to connect the nodes (steps in order)
#     workflow.add_edge(START, "generate_basic_description")
#     workflow.add_edge("generate_basic_description", "add_features_benefits")
#     workflow.add_edge("add_features_benefits", "create_marketing_message")
#     workflow.add_edge("create_marketing_message", "polish_final_description")
#     workflow.add_edge("polish_final_description", END)

# --- Workflow builder ---
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

    # Parallel branches
    workflow.add_edge("add_features_benefits", "create_marketing_message_premium")
    workflow.add_edge("add_features_benefits", "create_marketing_message_loyalty")
    workflow.add_edge("add_features_benefits", "create_marketing_message_corporate")

    # All branches join into the final polishing step
    workflow.add_edge("create_marketing_message_premium", "polish_final_description")
    workflow.add_edge("create_marketing_message_loyalty", "polish_final_description")
    workflow.add_edge("create_marketing_message_corporate", "polish_final_description")

    workflow.add_edge("polish_final_description", END)
       
    # Compile the workflow into a chain of actions
    chain = workflow.compile()

    return chain

# Step 8: Function to visualize the workflow (saved as an image)
# --- Visualization function ---
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
        ("create_marketing_message_corporate", "polish_final_description")
    ]
    graph.add_edges_from(edges)
    plt.figure(figsize=(10, 6))
    nx.draw(graph, with_labels=True, node_color='lightgreen', node_size=2000, font_size=10, font_weight='bold', edge_color='gray')
    plt.savefig("workflow.png")

# Main Streamlit function
def run_streamlit_app():
    st.title("Product Description Generator with Branching")
    product_name = st.text_input("Enter the product name:")

    if st.button("Generate Product Description"):
        state = State(
            product_name=product_name,
            basic_description="",
            features_benefits="",
            marketing_message_premium="",
            marketing_message_loyalty="",
            marketing_message_corporate="",
            final_description=""
        )

        chain = build_workflow()
        result = chain.invoke(state)

        st.subheader("Basic Description")
        st.write(result["basic_description"])

        st.subheader("Features & Benefits")
        st.write(result["features_benefits"])

        st.subheader("Premium Marketing Message")
        st.write(result["marketing_message_premium"])

        st.subheader("Loyalty Marketing Message")
        st.write(result["marketing_message_loyalty"])

        st.subheader("Corporate Marketing Message")
        st.write(result["marketing_message_corporate"])

        st.subheader("Final Description")
        st.write(result["final_description"])

        visualize_workflow()
        st.image("workflow.png", caption="Workflow Visualization")

if __name__ == "__main__":
    run_streamlit_app()
