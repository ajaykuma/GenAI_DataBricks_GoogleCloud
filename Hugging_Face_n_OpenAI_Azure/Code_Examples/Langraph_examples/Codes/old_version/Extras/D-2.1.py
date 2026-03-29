#Integrating with memory
import openai
from typing_extensions import TypedDict
from typing import List, Union
from langchain.schema import BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
#from dotenv import load_dotenv
import os
from dataclasses import asdict, dataclass
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st


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
    memory: Union[List[BaseMessage], None]  # Optional message history

# Initialize shared memory and LLM
def init_conversation():
    memory = ConversationBufferWindowMemory(k=5, return_messages=True)
    llm = AzureChatOpenAI(
        section..
    )
    return ConversationChain(llm=llm, memory=memory)
    
conversation = init_conversation()
memory = conversation.memory  # if you need direct access

# Flag to toggle features & benefits node mode
USE_MEMORY_FOR_FEATURES = True

def generate_basic_description(state: State):
    prompt = f"Write a brief description of a product named '{state['product_name']}'."
    output = conversation.run(prompt)
    return {"basic_description": output}

# Node: Features & Benefits WITH memory (conversation.run)
def add_features_benefits_with_memory(state: State):
    prompt = f"List key features and benefits of the product: {state['basic_description']}"
    output = conversation.run(prompt)
    return {"features_benefits": output}

# Node: Features & Benefits WITHOUT memory (direct raw client call)
def add_features_benefits_without_memory(state: State):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"List key features and benefits of the product: {state['basic_description']}"}
        ]
    )
    return {"features_benefits": response.choices[0].message.content, "memory": None}

# Wrapper node: Chooses which version to run based on flag
def add_features_benefits(state: State):
    if USE_MEMORY_FOR_FEATURES:
        return add_features_benefits_with_memory(state)
    else:
        return add_features_benefits_without_memory(state)

# Marketing message nodes (all with memory)
def create_marketing_message_premium(state: State):
    prompt = f"Create a premium marketing message for the product: {state['features_benefits']}"
    output = conversation.run(prompt)
    return {"marketing_message_premium": output}

def create_marketing_message_loyalty(state: State):
    prompt = f"Create a loyalty-based marketing message for the product: {state['features_benefits']}"
    output = conversation.run(prompt)
    return {"marketing_message_loyalty": output}

def create_marketing_message_corporate(state: State):
    prompt = f"Create a corporate marketing message for the product: {state['features_benefits']}"
    output = conversation.run(prompt)
    return {"marketing_message_corporate": output}

# Final polishing node (with memory)
def polish_final_description(state: State):
    prompt = (
        f"Combine and polish the product description including marketing messages:\n"
        f"Premium: {state['marketing_message_premium']}\n"
        f"Loyalty: {state['marketing_message_loyalty']}\n"
        f"Corporate: {state['marketing_message_corporate']}"
    )
    output = conversation.run(prompt)
    return {"final_description": output}

# --- Build the LangGraph workflow ---
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

    # Parallel branches after features & benefits
    workflow.add_edge("add_features_benefits", "create_marketing_message_premium")
    workflow.add_edge("add_features_benefits", "create_marketing_message_loyalty")
    workflow.add_edge("add_features_benefits", "create_marketing_message_corporate")

    # Join all branches to final polishing
    workflow.add_edge("create_marketing_message_premium", "polish_final_description")
    workflow.add_edge("create_marketing_message_loyalty", "polish_final_description")
    workflow.add_edge("create_marketing_message_corporate", "polish_final_description")

    workflow.add_edge("polish_final_description", END)

    return workflow.compile()

# --- Visualization (optional) ---
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
    nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, font_weight='bold')
    plt.savefig("workflow.png")

# --- Streamlit app ---
def run_streamlit_app():
    st.title("Product Description Generator with Memory Toggle")

    product_name = st.text_input("Enter the product name:")
    use_memory = st.checkbox("Use memory for Features & Benefits node", value=True)

    if st.button("Generate Product Description"):
        global USE_MEMORY_FOR_FEATURES
        USE_MEMORY_FOR_FEATURES = use_memory

        initial_state = State(
            product_name=product_name,
            basic_description="",
            features_benefits="",
            marketing_message_premium="",
            marketing_message_loyalty="",
            marketing_message_corporate="",
            final_description="",
            memory=[]
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

if __name__ == "__main__":
    run_streamlit_app()
