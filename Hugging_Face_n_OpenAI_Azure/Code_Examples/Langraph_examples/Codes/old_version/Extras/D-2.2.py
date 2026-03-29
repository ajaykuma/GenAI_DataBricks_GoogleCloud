#Integrating with memory
import openai
from typing_extensions import TypedDict
from typing import List, Union
from langchain.schema import BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from pydantic import PrivateAttr
from difflib import SequenceMatcher

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

# --- Tracking Memory ---
class TrackingConversationBufferWindowMemory(ConversationBufferWindowMemory):
    _similar_query_counts: dict = PrivateAttr(default_factory=dict)

    def track_query_usage(self, new_prompt: str):
        # Check for similar queries in memory
        found_similar = False
        for existing_prompt in self._similar_query_counts:
            similarity = SequenceMatcher(None, new_prompt, existing_prompt).ratio()
            if similarity > 0.7:
                self._similar_query_counts[existing_prompt] += 1
                found_similar = True
                break
        if not found_similar:
            self._similar_query_counts[new_prompt] = 1

    def get_query_usage_summary(self) -> str:
        if not self._similar_query_counts:
            return "[no repeated queries yet]"
        return "\n".join(
            [f"Seen similar to: '{q}' → {count} time(s)" for q, count in self._similar_query_counts.items()]
        )
        
# Initialize shared memory and LLM
def init_conversation():
    memory = TrackingConversationBufferWindowMemory(k=5, return_messages=True)
    llm = AzureChatOpenAI(
    section..
    )
    return ConversationChain(llm=llm, memory=memory)
    
conversation = init_conversation()
memory = conversation.memory  # if you need direct access

# Flag to toggle features & benefits node mode
USE_MEMORY_FOR_FEATURES = True

# --- Nodes ---
def generate_basic_description(state: State):
    prompt = f"Write a brief description of a product named '{state['product_name']}'."
    memory.track_query_usage(prompt)
    output = conversation.run(prompt)
    return {"basic_description": output}

def add_features_benefits_with_memory(state: State):
    prompt = f"List key features and benefits of the product: {state['basic_description']}"
    memory.track_query_usage(prompt)
    output = conversation.run(prompt)
    return {"features_benefits": output}

def add_features_benefits_without_memory(state: State):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"List key features and benefits of the product: {state['basic_description']}"}]
    )
    return {"features_benefits": response.choices[0].message.content, "memory": None}

def add_features_benefits(state: State):
    if USE_MEMORY_FOR_FEATURES:
        return add_features_benefits_with_memory(state)
    else:
        return add_features_benefits_without_memory(state)

def create_marketing_message_premium(state: State):
    prompt = f"Create a premium marketing message for the product: {state['features_benefits']}"
    memory.track_query_usage(prompt)
    output = conversation.run(prompt)
    return {"marketing_message_premium": output}

def create_marketing_message_loyalty(state: State):
    prompt = f"Create a loyalty-based marketing message for the product: {state['features_benefits']}"
    memory.track_query_usage(prompt)
    output = conversation.run(prompt)
    return {"marketing_message_loyalty": output}

def create_marketing_message_corporate(state: State):
    prompt = f"Create a corporate marketing message for the product: {state['features_benefits']}"
    memory.track_query_usage(prompt)
    output = conversation.run(prompt)
    return {"marketing_message_corporate": output}

def polish_final_description(state: State):
    prompt = (
        f"Combine and polish the product description including marketing messages:\n"
        f"Premium: {state['marketing_message_premium']}\n"
        f"Loyalty: {state['marketing_message_loyalty']}\n"
        f"Corporate: {state['marketing_message_corporate']}"
    )
    memory.track_query_usage(prompt)
    output = conversation.run(prompt)
    return {"final_description": output}

# --- LangGraph Workflow ---
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

# --- Visualize (optional) ---
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

# --- Streamlit UI ---
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

        st.subheader("Memory Usage Summary:")
        st.text(memory.get_query_usage_summary())

        visualize_workflow()
        st.image("workflow.png", caption="Workflow")

# --- Run app ---
if __name__ == "__main__":
    run_streamlit_app()
