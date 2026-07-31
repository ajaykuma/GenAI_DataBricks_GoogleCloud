#Fixing issues seen in previous i.e. Demo2.1.py
#Memory aware responses
    #So issue is recurring again should become > The brightness issue you mentioned earlier is recurring...
#Strict domain control (IT support only)
    #no balance in my savings account should > I can only assist with IT support issues...
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

        # Clean messages
        formatted_messages = []
        for m in messages:
            content = m.get("content", "").strip()
            role = m.get("role", "user")
            if content:
                formatted_messages.append({"role": role, "content": content})

        if not formatted_messages:
            formatted_messages = [{"role": "user", "content": "Hello"}]

        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=formatted_messages
        )

        return response.choices[0].message.content


# -----------------------------
# Initialize Agents
# -----------------------------
if "initialized" not in st.session_state:

    st.session_state.support_agent = AzureAgent(
        name="support_agent",
        system_message="""
You are an IT Support Assistant.

STRICT RULES:
1. ONLY answer IT / technical support questions.
2. If user asks NON-IT question → politely refuse.
3. Always use conversation history for context.
4. If user says "issue again" → refer to previous issue.
5. Give specific troubleshooting, NOT generic replies.
6. If unclear → ask targeted follow-up questions.

Examples of NON-IT:
- Banking
- Finance
- Personal advice

Response style:
- Clear
- Context-aware
- Professional
"""
    )

    st.session_state.chat_history = []
    st.session_state.initialized = True


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Multi-Agent IT Support Chatbot")
st.write("Context-aware + Domain-restricted AutoGen System")

user_input = st.text_input("Enter your issue:")

# -----------------------------
# Helper: Build conversation memory
# -----------------------------
def build_messages(user_input):

    messages = [
        {
            "role": "system",
            "content": st.session_state.support_agent.system_message
        }
    ]

    for role, msg in st.session_state.chat_history:
        if role == "Customer":
            messages.append({"role": "user", "content": msg})
        else:
            messages.append({"role": "assistant", "content": msg})

    messages.append({"role": "user", "content": user_input})

    return messages


# -----------------------------
# Domain filter (IMPORTANT)
# -----------------------------
def is_it_related(text):

    keywords = [
        "computer", "laptop", "screen", "display", "keyboard",
        "mouse", "wifi", "internet", "software", "hardware",
        "windows", "mac", "linux", "error", "system", "desktop",
        "printer", "network", "brightness", "issue", "problem"
    ]

    text = text.lower()

    # Check current message
    if any(word in text for word in keywords):
        return True

    # Check previous conversation context
    for role, msg in reversed(st.session_state.chat_history):
        msg = msg.lower()
        if any(word in msg for word in keywords):
            return True

    return False

# -----------------------------
# Handle conversation
# -----------------------------
if st.button("Send"):

    if not user_input.strip():
        st.warning("Please enter a message")
        st.stop()

    st.session_state.chat_history.append(("Customer", user_input))

    # DOMAIN CONTROL
    if not is_it_related(user_input):
        response = (
            "I can only assist with IT support-related issues "
            "(e.g., computers, software, networks). "
            "Please ask a technical question."
        )

    else:
        messages = build_messages(user_input)

        response = st.session_state.support_agent.generate_reply(
            messages=messages,
            sender="customer"
        )

    st.session_state.chat_history.append(("Support", response))


# -----------------------------
# Display chat
# -----------------------------
st.subheader("Conversation")

for role, msg in st.session_state.chat_history:
    if role == "Customer":
        st.markdown(f"**Customer:** {msg}")
    else:
        st.markdown(f"**Support:** {msg}")

