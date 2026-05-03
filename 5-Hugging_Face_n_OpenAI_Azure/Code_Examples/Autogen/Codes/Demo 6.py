import streamlit as st
import autogen
import openai
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

# Step 2: Define the Diagnostic Agent
class DiagnosticAgent(autogen.AssistantAgent):
    def __init__(self, name="DiagnosticAgent"):
        """
        Initializes the Diagnostic Agent.
        - Identifies the issue based on user input.
        - Determines whether to resolve or delegate to the Resolution Agent.
        """
        super().__init__(name=name)

    def analyze_issue(self, user_message):
        """
        Determines whether the issue is simple or complex.
        Returns a structured report for the Resolution Agent.
        """
        simple_issues = ["password reset", "slow internet", "software update", "printer not working"]
        for issue in simple_issues:
            if issue in user_message.lower():
                return {"type": "simple", "issue": user_message}

        return {"type": "complex", "issue": user_message}

# Step 3: Define the Resolution Agent
class ResolutionAgent(autogen.AssistantAgent):
    def __init__(self, name="ResolutionAgent", model="gpt-4.1"):
        """
        Initializes the Resolution Agent.
        - Handles issue resolution based on task delegation.
        - Escalates cases if needed.
        """
        super().__init__(name=name)
        self.model = model

    def resolve_issue(self, task):
        """
        Resolves the issue if simple, otherwise provides advanced troubleshooting.
        """
        if task["type"] == "simple":
            return f"The issue '{task['issue']}' is resolved with basic troubleshooting."
        else:
            return self._advanced_troubleshooting(task["issue"])

    def _advanced_troubleshooting(self, issue):
        """
        Uses GPT to generate advanced troubleshooting steps.
        """
        prompt = f"""
        The user reported an IT issue: "{issue}". 
        You are an IT Resolution Agent. Provide advanced troubleshooting steps.
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an IT expert specializing in troubleshooting."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

# Step 4: Initialize the Agents
diagnostic_agent = DiagnosticAgent()
resolution_agent = ResolutionAgent()

# Step 5: Deploy the Agents Using Streamlit
st.title("AI-Powered IT Support: Task Delegation & Decision Making")
st.write("Enter an IT issue, and the AI agents will determine whether to resolve it directly or escalate it.")

user_input = st.text_area("Describe your IT issue:")

if st.button("Submit Issue"):
    if user_input.strip():
        # Diagnostic Agent analyzes the issue
        task = diagnostic_agent.analyze_issue(user_input)

        # Resolution Agent resolves or escalates
        response = resolution_agent.resolve_issue(task)

        st.subheader("AI Response:")
        st.write(response)
    else:
        st.warning("Please enter a valid IT issue.")
