# Step 1: Set Up API Key and Environment
import os
from dotenv import load_dotenv
import autogen
from openai import AzureOpenAI, APIError  # Use AzureOpenAI directly

# Load environment variables from .env
#Remember to place .env file in the Codes folder
load_dotenv()

# Step 1: Load environment variables (API keys)
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION")
)

# Step 2: Create Customer Agent
customer_agent = autogen.UserProxyAgent(
    name="customer",
    human_input_mode="ALWAYS",  # Allows manual input
    code_execution_config={"use_docker": False},
    max_consecutive_auto_reply=5
)

# Step 3: Create Support Agent using Azure OpenAI
#Telling autogen that handle Azure calls internally, thus commenting out this section

# support_agent = autogen.AssistantAgent(
#     name="support_agent",
#     llm_config={
#         "config_list": [
#             {
#                 "api_type": "azure",
#                 "api_key": os.getenv("API_KEY"),
#                 "api_version": os.getenv("AZURE_API_VERSION"),
#                 "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
#                 "model": "gpt-4.1"
#             }
#         ],
#         "temperature": 0.7,
#     },
#     system_message="You are a helpful AI support agent. Answer customer queries clearly and professionally.",
#     code_execution_config={"use_docker": False},
#     max_consecutive_auto_reply=5
# )

# Step 3: Create Support Agent using Azure OpenAI
#Keep AutoGen for chat orchestration
#But use your working Azure client for actual API calls
class SupportAgent(autogen.AssistantAgent):
    def generate_reply(self, messages=None, sender=None, **kwargs):
        # Convert AutoGen messages → OpenAI format
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

        response = client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT_NAME"),
            messages=formatted_messages
        )
        return response.choices[0].message.content

support_agent = SupportAgent(
    name="support_agent",
    system_message="You are a helpful AI support agent. Answer clearly and professionally."
)

# Step 4: Run a simulated customer interaction safely
try:
    customer_agent.initiate_chat(support_agent, message="I need help tracking my order.")
except APIError as e:
    print("API error:", e)
except Exception as e:
    print("Unexpected error:", e)
