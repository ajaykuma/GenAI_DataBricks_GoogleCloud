import os
import streamlit as st
import autogen
import networkx as nx
import matplotlib.pyplot as plt

AZURE_API_KEY = "mykey"  # <- Put your Azure OpenAI key here

# ====== Create Support Agent ======
support_agent = autogen.AssistantAgent(
    name="support_agent",
    llm_config={
        "config_list": [
            {
                "api_type": "azure",
                "azure_endpoint": "https://info-mdub1hyj-eastus2.cognitiveservices.azure.com/",
                "api_version": "2023-12-01-preview",
                "api_key": AZURE_API_KEY,
                "model": "gpt-4o-mini",
            }
        ],
        "temperature": 0.7,
    },
    system_message="You are a helpful AI support agent. Answer customer queries clearly and professionally.",
    code_execution_config={'use_docker': False}
)

# ====== Visualization Function ======
def visualize_conversation(messages):
    graph = nx.DiGraph()
    for i in range(len(messages) - 1):
        sender = messages[i][0]
        receiver = messages[i + 1][0]
        graph.add_edge(sender, receiver)

    plt.figure(figsize=(3, 2))  # smaller size for side display
    nx.draw(graph, with_labels=True, node_color="lightblue", edge_color="gray",
            node_size=1500, font_size=8, font_weight="bold", arrows=True)
    plt.title("Flow", fontsize=10)
    plt.tight_layout()
    plt.savefig("conversation_flow.png")
    plt.close()

# ====== Streamlit UI ======
st.set_page_config(page_title="Autogen Support Chat", page_icon="💬", layout="wide")
st.title("💬 AI Support Chat with Visualization")

# Chat input MUST be outside of columns
user_input = st.chat_input("Type your question...")

# Layout: Chat on left, graph on right
col1, col2 = st.columns([0.7, 0.3])

with col1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for role, content in st.session_state.messages:
        if role == "customer":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

    # If user typed something
    if user_input:
        st.session_state.messages.append(("customer", user_input))
        st.chat_message("user").write(user_input)

        reply_text = support_agent.generate_reply(messages=[{"role": "user", "content": user_input}])
        if isinstance(reply_text, list):
            reply_text = reply_text[0]

        st.session_state.messages.append(("support_agent", reply_text))
        st.chat_message("assistant").write(reply_text)

        visualize_conversation(st.session_state.messages)

with col2:
    if os.path.exists("conversation_flow.png"):
        st.image("conversation_flow.png", caption="Conversation Flow", use_column_width=True)
