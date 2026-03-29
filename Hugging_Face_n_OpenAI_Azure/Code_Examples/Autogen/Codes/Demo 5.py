import streamlit as st
import autogen
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: Load environment variables (API keys)
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION")
)

# Step 1: Define the Diagnostic Agent
class DiagnosticAgent(autogen.AssistantAgent):
    def __init__(self, name="DiagnosticAgent", model="gpt-4.1"):
        super().__init__(name=name)
        self.model = model
    
    def diagnose_issue(self, message):
        """
        Step 2: The Diagnostic Agent analyzes the issue and determines possible causes.
        """
        prompt = f"""
        The user has reported an IT issue: "{message}".
        As the Diagnostic Agent, analyze the problem and list possible causes.
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an IT diagnostic expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

# Step 3: Define the Resolution Agent
class ResolutionAgent(autogen.AssistantAgent):
    def __init__(self, name="ResolutionAgent", model="gpt-4.1"):
        super().__init__(name=name)
        self.model = model
    
    def provide_solution(self, diagnosis):
        """
        Step 4: The Resolution Agent suggests a fix based on the diagnosis from the Diagnostic Agent.
        """
        prompt = f"""
        The Diagnostic Agent has identified the following possible causes:
        "{diagnosis}".
        As the Resolution Agent, suggest step-by-step troubleshooting solutions.
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an IT troubleshooting expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

# Step 5: Deploy via Streamlit
st.title("AutoGen IT Support Chatbot - Agent Collaboration")
st.write("An AI-powered chatbot where agents collaborate to diagnose and resolve IT issues.")

# User Input
user_input = st.text_area("Describe your IT issue:")
if st.button("Get Support"):
    if user_input.strip():
        diagnostic_agent = DiagnosticAgent()
        resolution_agent = ResolutionAgent()
        
        diagnosis = diagnostic_agent.diagnose_issue(user_input)
        solution = resolution_agent.provide_solution(diagnosis)
        
        st.subheader("Diagnosis:")
        st.write(diagnosis)
        
        st.subheader("Suggested Solution:")
        st.write(solution)
    else:
        st.warning("Please enter a valid IT issue.")


