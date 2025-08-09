import autogen
import openai
import streamlit as st
from dotenv import load_dotenv
import os

client = openai.AzureOpenAI(
    api_key="mykey",    
    api_version="2023-12-01-preview",
    azure_endpoint="https://info-mdub1hyj-eastus2.cognitiveservices.azure.com/"
)

# ====== Bot Class ======
class MultiStepITSupportBot(autogen.AssistantAgent):
    def __init__(self, name, model="gpt-4o-mini"):
        super().__init__(name=name)
        self.model = model

    def generate_reply(self, message, previous_steps=[]):
        prompt = f"""
        The user reported an IT issue: "{message}".
        Steps attempted so far: {previous_steps}
        Your task is to:
        1. Identify the next possible cause of the issue.
        2. Provide the next step in a step-by-step troubleshooting guide.
        3. End with exactly: 'Did this step solve your issue? (yes/no)'
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an IT support assistant that provides structured, step-by-step troubleshooting."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()


# ====== Streamlit UI ======
st.set_page_config(page_title="Multi-Step IT Support Chatbot", layout="wide")

st.title("Multi-Step IT Support Chatbot")
st.write("Describe your IT issue, and our assistant will walk you through troubleshooting, step by step.")

# Initialize session state
if "bot" not in st.session_state:
    st.session_state.bot = MultiStepITSupportBot(name="StepByStepHelpBot")

if "issue" not in st.session_state:
    st.session_state.issue = ""

if "previous_steps" not in st.session_state:
    st.session_state.previous_steps = []

if "current_step" not in st.session_state:
    st.session_state.current_step = ""


# ====== Layout: Chat (left) | History (right) ======
col1, col2 = st.columns([0.7, 0.3])

with col1:
    # Step 1: Enter issue
    if not st.session_state.issue:
        st.session_state.issue = st.text_input("Enter your IT issue:")

    # Step 2: Get first step
    if st.button("Get Help") and st.session_state.issue.strip():
        st.session_state.current_step = st.session_state.bot.generate_reply(
            st.session_state.issue, st.session_state.previous_steps
        )

    # Step 3: Show current step
    if st.session_state.current_step:
        st.subheader("Next Step:")
        st.write(st.session_state.current_step)

        # Ask Yes/No directly below current step
        user_feedback = st.radio(
            "Did this step solve your issue?", ("", "Yes", "No"),
            key=f"feedback_{len(st.session_state.previous_steps)}"
        )

        if user_feedback == "Yes":
            st.success("Great! Your issue is resolved.")
            st.session_state.issue = ""
            st.session_state.previous_steps = []
            st.session_state.current_step = ""

        elif user_feedback == "No":
            st.session_state.previous_steps.append(st.session_state.current_step)
            st.session_state.current_step = st.session_state.bot.generate_reply(
                st.session_state.issue, st.session_state.previous_steps
            )
            st.experimental_rerun()


with col2:
    st.subheader("Troubleshooting History")
    if st.session_state.previous_steps:
        for i, step in enumerate(st.session_state.previous_steps, start=1):
            st.markdown(f"**Step {i}:** {step}")
    else:
        st.info("No previous steps yet. Start troubleshooting to see history here.")
