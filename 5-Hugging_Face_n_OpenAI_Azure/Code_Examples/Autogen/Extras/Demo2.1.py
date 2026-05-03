#Building a controlled AutoGen chatbot (single-turn per input) with memory stored in Streamlit.

import os
from dotenv import load_dotenv
import autogen
from openai import AzureOpenAI
import streamlit as st

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION")
)

DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")

# -----------------------------
# Custom Azure Agent
# -----------------------------
class AzureAgent(autogen.AssistantAgent):
    def generate_reply(self, messages=None, sender=None, **kwargs):

        formatted_messages = []

        for m in messages:
            content = m.get("content", "").strip()
            role = m.get("role", "user")

            # Skip empty messages
            if content:
                formatted_messages.append({
                    "role": role,
                    "content": content
                })

        #ensure at least one message
        if not formatted_messages:
            formatted_messages = [
                {"role": "user", "content": "Hello"}
            ]

        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=formatted_messages
        )

        return response.choices[0].message.content

# -----------------------------
# Initialize Agents (once)
# -----------------------------
if "agents_initialized" not in st.session_state:

    st.session_state.support_agent = AzureAgent(
        name="support_agent",
        system_message=(
            "You are a helpful customer support agent. "
            "Answer clearly. If the issue is complex, say 'ESCALATE'."
        )
    )

    st.session_state.escalation_agent = AzureAgent(
        name="escalation_agent",
        system_message=(
            "You are a senior support specialist. "
            "Handle escalated issues with detailed solutions."
        )
    )

    st.session_state.customer_agent = autogen.UserProxyAgent(
        name="customer",
        human_input_mode="NEVER",  # Streamlit handles input
        code_execution_config={"use_docker": False}
    )

    st.session_state.chat_history = []
    st.session_state.agents_initialized = True

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Multi-Agent Support Chatbot")
st.write("Customer → Support → Escalation (AutoGen System)")

user_input = st.text_input("Enter your issue:")

# -----------------------------
# Handle conversation
# -----------------------------

if st.button("Send"):

    if not user_input.strip():
        st.warning("Please enter a message")
        st.stop()

    # Step 1: Customer → Support (DIRECT CALL, no loop)
    support_msg = st.session_state.support_agent.generate_reply(
        messages=[{"role": "user", "content": user_input}],
        sender="customer"
    )

    st.session_state.chat_history.append(("Customer", user_input))
    st.session_state.chat_history.append(("Support", support_msg))

    # Step 2: Escalation check
    if "ESCALATE" in support_msg.upper():

        escalation_msg = st.session_state.escalation_agent.generate_reply(
            messages=[{"role": "user", "content": user_input}],
            sender="support_agent"
        )

        st.session_state.chat_history.append(("Escalation", escalation_msg))


# -----------------------------
# Display chat history
# -----------------------------
st.subheader("Conversation")

for role, msg in st.session_state.chat_history:
    if role == "Customer":
        st.markdown(f"** {role}:** {msg}")
    elif role == "Support":
        st.markdown(f"** {role}:** {msg}")
    else:
        st.markdown(f"** {role}:** {msg}")
