#Multiple Agents : Each has a role (Diagnose → Fix → Validate)
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
# Custom Azure AutoGen Agent
# -----------------------------
class AzureAgent(autogen.AssistantAgent):
    def __init__(self, name, system_message):
        super().__init__(name=name, system_message=system_message)

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
            temperature=0.4
        )

        return response.choices[0].message.content.strip()


# -----------------------------
# Initialize Session State
# -----------------------------
if "initialized" not in st.session_state:

    # Agents
    st.session_state.diagnoser = AzureAgent(
        name="Diagnoser",
        system_message="""
You are an IT Issue Diagnoser.

- Identify root cause
- List possible causes
- Be structured and concise
"""
    )

    st.session_state.troubleshooter = AzureAgent(
        name="Troubleshooter",
        system_message="""
You are an IT Troubleshooting Expert.

- Provide step-by-step fixes
- Keep instructions simple and actionable
"""
    )

    st.session_state.validator = AzureAgent(
        name="Validator",
        system_message="""
You are a QA Validator.

- Check if solution is complete
- Identify missing steps or risks
- Improve solution if needed
"""
    )

    st.session_state.chat_history = []

    st.session_state.issue_state = {
        "type": None,
        "status": None
    }

    st.session_state.initialized = True


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Multi-Agent IT Support System (Azure Powered)")
st.write("Structured Memory + Multi-Agent Reasoning")

user_input = st.text_input("Enter your issue:")


# -----------------------------
# Issue Tracking
# -----------------------------
def update_issue_tracking(user_input):

    text = user_input.lower()
    current_type = st.session_state.issue_state["type"]

    if "screen" in text or "display" in text:
        issue_type = "screen issue"
    elif "keyboard" in text:
        issue_type = "keyboard issue"
    elif "printer" in text:
        issue_type = "printer issue"
    else:
        issue_type = current_type

    if "resolved" in text or "fixed" in text:
        status = "resolved"
    elif "again" in text or "still" in text:
        status = "recurring"
    else:
        status = "active"

    st.session_state.issue_state["type"] = issue_type
    st.session_state.issue_state["status"] = status


# -----------------------------
# Domain Filter
# -----------------------------
def is_it_related(text):

    keywords = [
        "computer", "laptop", "screen", "keyboard", "mouse",
        "wifi", "internet", "software", "hardware",
        "windows", "mac", "linux", "printer", "network"
    ]

    text = text.lower()

    if any(word in text for word in keywords):
        return True

    for role, msg in reversed(st.session_state.chat_history):
        if any(word in msg.lower() for word in keywords):
            return True

    return False


# -----------------------------
# Build Messages
# -----------------------------
def build_messages(user_input):

    history_text = "\n".join(
        [f"{role}: {msg}" for role, msg in st.session_state.chat_history[-6:]]
    )

    issue_type = st.session_state.issue_state["type"]
    issue_status = st.session_state.issue_state["status"]

    return [
        {
            "role": "system",
            "content": f"""
Current issue: {issue_type}
Issue status: {issue_status}

Conversation history:
{history_text}
"""
        },
        {"role": "user", "content": user_input}
    ]


# -----------------------------
# Multi-Agent Pipeline
# -----------------------------
def run_agents(user_input):

    base_messages = build_messages(user_input)

    # Step 1: Diagnosis
    diagnosis = st.session_state.diagnoser.generate_reply(base_messages)

    # Step 2: Troubleshooting
    solution = st.session_state.troubleshooter.generate_reply([
        {"role": "user", "content": f"Diagnosis:\n{diagnosis}"}
    ])

    # Step 3: Validation
    validation = st.session_state.validator.generate_reply([
        {"role": "user", "content": f"Solution:\n{solution}"}
    ])

    final_response = f"""
Diagnosis:
{diagnosis}

Solution:
{solution}

Validation:
{validation}
"""

    return final_response


# -----------------------------
# Handle Conversation
# -----------------------------
if st.button("Send"):

    if not user_input.strip():
        st.warning("Please enter a message")
        st.stop()

    st.session_state.chat_history.append(("Customer", user_input))

    if not is_it_related(user_input):
        response = "I only handle IT-related issues."
    else:
        update_issue_tracking(user_input)
        response = run_agents(user_input)

    st.session_state.chat_history.append(("Support", response))


# -----------------------------
# Display Chat
# -----------------------------
st.subheader("Conversation")

for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}:** {msg}")
