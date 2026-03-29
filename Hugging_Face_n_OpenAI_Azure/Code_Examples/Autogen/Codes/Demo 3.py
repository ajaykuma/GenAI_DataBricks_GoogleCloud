import autogen
from openai import AzureOpenAI
import streamlit as st
from dotenv import load_dotenv
import os

#load_dotenv("E:\\Lesson_3_demos\\.env")
load_dotenv()

# Step 1: Load environment variables (API keys)
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION")
)

# Step 2: Define an IT Support chatbot with multi-step reasoning
class MultiStepITSupportBot(autogen.AssistantAgent):
    def __init__(self, name, model="gpt-4.1"):
        """
        Initializes the chatbot with:
        - A name for identification.
        - A selected GPT model (default: GPT-3.5 Turbo).
        """
        super().__init__(name=name)
        self.model = model  # Store the selected model

    def generate_reply(self, message, previous_steps=[]):
        """
       Handles user queries using multi-step reasoning:
        - Understand the issue.
        - Identify possible causes.
        - Provide a step-by-step resolution plan.
        - Ask for confirmation before proceeding to the next step.
        """
        response = self._get_gpt_response(message, previous_steps)
        return response

    def _get_gpt_response(self, message, previous_steps):
        """
        Uses OpenAI's GPT to break down the problem logically.
        - Identifies potential causes.
        - Provides step-by-step troubleshooting.
        - Asks the user for confirmation before moving to the next step.
        """

        prompt = f"""
        The user reported an IT issue: "{message}". 
        Steps attempted so far: {previous_steps}
        Your task is to:
        1. Identify the next possible cause of the issue.
        2. Provide the next step in a step-by-step troubleshooting guide.
        3. Ask: "Did this step solve your issue? (yes/no)"
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an IT support assistant that provides structured, step-by-step troubleshooting."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

'''
> Azure OpenAI
Handles:
Prompt processing
Reasoning
Response generation
> AutoGen (in our code)
NOT doing:
Agent orchestration
Multi-agent communication
Tool usage
Memory handling

So, It’s basically just acting as a wrapper class / label, not as a framework.

'''
# Step 3: Streamlit Web App for Multi-Step IT Support Bot
st.title("Multi-Step IT Support Chatbot")
st.write("Describe your IT issue, and our assistant will provide a step-by-step troubleshooting guide.")

# Initialize chatbot and session state
if "bot" not in st.session_state:
    st.session_state.bot = MultiStepITSupportBot(name="StepByStepHelpBot")

if "issue" not in st.session_state:
    st.session_state.issue = ""

if "previous_steps" not in st.session_state:
    st.session_state.previous_steps = []

# Step 4: User Input
if not st.session_state.issue:
    st.session_state.issue = st.text_input("Enter your IT issue:")

if st.button("Get Help"):
    if st.session_state.issue.strip():  # Ensure input is not empty
        response = st.session_state.bot.generate_reply(st.session_state.issue, st.session_state.previous_steps)
        st.session_state.previous_steps.append(response)
        st.subheader("Next Step:")
        st.write(response)
    else:
        st.warning("Please enter an IT issue before submitting.")

# Step 4.2: User Confirmation
if st.session_state.previous_steps:
    user_feedback = st.radio("Did this step solve your issue?", ("", "Yes", "No"))
    if user_feedback == "Yes":
        st.success("Great! Your issue is resolved.")
        st.session_state.issue = ""
        st.session_state.previous_steps = []
    elif user_feedback == "No":
        response = st.session_state.bot.generate_reply(st.session_state.issue, st.session_state.previous_steps)
        st.session_state.previous_steps.append(response)
        st.subheader("Next Step:")
        st.write(response)
