'''
AutoGen normally:

> manages conversations between agents
> handles message passing
> calls LLM internally via llm_config

But in our code:
We are NOT using:

-- llm_config
-- agent-to-agent chat
-- AutoGen orchestration
-- AutoGen memory system

Summary:
We are using AutoGen > But only as a base class — not as a framework
Our chatbot is essentially:

Streamlit + Azure OpenAI + custom Python lo
'''

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

#We are using AutoGen here but we override its core behaviorin generate_reply below
# Step 2: Define an IT Support chatbot that remembers past issues
class ITSupportBot(autogen.AssistantAgent):
    def __init__(self, name, memory=None, model="gpt-4.1"):  
        super().__init__(name=name)
        self.memory = memory if memory is not None else {}  # Stores past user issues
        self.model = model  # Specifies GPT model

    def generate_reply(self, message, sender, **kwargs):
        """
        Step 3: Generates a response based on user input and past issues.
        - Retrieves past issues if available
        - Stores the latest issue in memory
        - Calls the GPT model to generate a response
        """
        context = self.memory.get(sender, "")  # Retrieves past issue if available
        self.memory[sender] = message  # Stores latest issue in memory
        
        response = self._get_gpt_response(message, context)  # Calls GPT for reply
        return response
    
    def _get_gpt_response(self, message, context):
        """
        Step 4: Calls OpenAI's GPT model to generate a response with past issue history.
        - Constructs a prompt including past conversation history
        - Sends the prompt to GPT-model..
        - Returns the generated response
        """
        prompt = f"User's previous issue: {context}\nNew issue: {message}\nIT Support Response:"
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful IT support assistant providing troubleshooting steps."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

# Step 5: Streamlit UI for real-time chatbot interaction
st.title("IT Support Chatbot")
st.write("Ask me about your IT issues, and I'll provide troubleshooting steps!")

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ITSupportBot(name="HelpDeskBot")

# Input field for user query
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = st.session_state.chatbot.generate_reply(user_input, "User1")
        st.write(f"**HelpDeskBot:** {response}")

#if st.button("reset session"):
#    for key in st.session_state.keys():
#        del st.session_state[key]
#    st.experimental_rerun()


