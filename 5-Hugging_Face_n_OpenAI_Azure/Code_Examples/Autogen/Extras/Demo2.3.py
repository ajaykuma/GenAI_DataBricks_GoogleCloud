import autogen
import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Azure OpenAI Client
# -----------------------------
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION")
)

DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")

# -----------------------------
# Custom AutoGen Azure Agent
# -----------------------------
class AzureAgent(autogen.AssistantAgent):
    def generate_reply(self, messages=None, sender=None, **kwargs):

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
            messages=formatted_messages,
            temperature=0.5
        )

        return response.choices[0].message.content.strip()


# -----------------------------
# Initialize Session State
# -----------------------------
if "initialized" not in st.session_state:

    st.session_state.support_agent = AzureAgent(
        name="support_agent",
        system_message="""
You are a professional IT Support Assistant.

STRICT RULES:
1. ONLY answer IT-related issues.
2. If user asks NON-IT → politely refuse.
3. Track issue lifecycle (active → resolved → recurring).
4. If issue is recurring, explicitly acknowledge it.
5. If new issue appears, handle it separately but remember previous.
6. Avoid generic replies — always refer to known context.
7. Provide step-by-step troubleshooting.

Response style:
- Clear
- Structured
- Context-aware
"""
    )

    st.session_state.chat_history = []

    # STRUCTURED MEMORY
    st.session_state.issue_state = {
        "type": None,
        "status": None   # active / resolved / recurring
    }

    st.session_state.initialized = True


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Multi-Agent IT Support Chatbot")
st.write("Context-aware + Structured Memory + Domain Control")

user_input = st.text_input("Enter your issue:")


# -----------------------------
# Issue Tracking (STRUCTURED)
# -----------------------------
def update_issue_tracking(user_input):

    text = user_input.lower()

    current_type = st.session_state.issue_state["type"]

    # Detect issue type
    if any(word in text for word in ["screen", "display"]):
        issue_type = "screen issue"

    elif any(word in text for word in ["keyboard", "keys"]):
        issue_type = "keyboard issue"

    elif "printer" in text:
        issue_type = "printer issue"

    else:
        issue_type = current_type

    # Detect lifecycle
    if any(word in text for word in ["fixed", "resolved"]):
        status = "resolved"

    elif any(word in text for word in ["again", "recurring", "still"]):
        status = "recurring"

    else:
        status = "active"

    st.session_state.issue_state["type"] = issue_type
    st.session_state.issue_state["status"] = status


# -----------------------------
# Domain Filter (Context-aware)
# -----------------------------
def is_it_related(text):

    keywords = [
        "computer", "laptop", "screen", "display", "keyboard",
        "mouse", "wifi", "internet", "software", "hardware",
        "windows", "mac", "linux", "error", "system", "desktop",
        "printer", "network", "brightness", "issue", "problem"
    ]

    text = text.lower()

    # Check current input
    if any(word in text for word in keywords):
        return True

    # Check previous conversation
    for role, msg in reversed(st.session_state.chat_history):
        msg = msg.lower()
        if any(word in msg for word in keywords):
            return True

    return False


# -----------------------------
# Build Messages (Context + Memory)
# -----------------------------
def build_messages(user_input):

    history_text = "\n".join(
        [f"{role}: {msg}" for role, msg in st.session_state.chat_history[-6:]]
    )

    issue_type = st.session_state.issue_state["type"]
    issue_status = st.session_state.issue_state["status"]

    messages = [
        {
            "role": "system",
            "content": f"""
{st.session_state.support_agent.system_message}

Current issue: {issue_type}
Issue status: {issue_status}

Conversation history:
{history_text}
"""
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
# Handle Conversation
# -----------------------------
if st.button("Send"):

    if not user_input.strip():
        st.warning("Please enter a message")
        st.stop()

    # Save user input
    st.session_state.chat_history.append(("Customer", user_input))

    # Domain restriction
    if not is_it_related(user_input):
        response = (
            "I can only assist with IT support-related issues "
            "(computers, software, networks). Please ask a relevant question."
        )

    else:
        # Update structured memory
        update_issue_tracking(user_input)

        messages = build_messages(user_input)

        response = st.session_state.support_agent.generate_reply(
            messages=messages,
            sender="customer"
        )

    # Save response
    st.session_state.chat_history.append(("Support", response))


# -----------------------------
# Display Chat
# -----------------------------
st.subheader("Conversation")

for role, msg in st.session_state.chat_history:
    if role == "Customer":
        st.markdown(f"**Customer:** {msg}")
    else:
        st.markdown(f"**Support:** {msg}")

