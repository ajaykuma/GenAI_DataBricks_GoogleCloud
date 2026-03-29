import os
from dotenv import load_dotenv
import autogen
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Azure client (your working setup)
client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION")
)

DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")

# -----------------------------
# Custom Base Agent (uses Azure)
# -----------------------------
class AzureAgent(autogen.AssistantAgent):
    def generate_reply(self, messages=None, sender=None, **kwargs):
        formatted_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=formatted_messages
        )

        return response.choices[0].message.content


# -----------------------------
# Support Agent
# -----------------------------
support_agent = AzureAgent(
    name="support_agent",
    system_message=(
        "You are a helpful customer support agent. "
        "Answer clearly. If the issue is complex or unresolved, say 'ESCALATE'."
    )
)

# -----------------------------
# Escalation Agent
# -----------------------------
escalation_agent = AzureAgent(
    name="escalation_agent",
    system_message=(
        "You are a senior support specialist. "
        "Handle complex or escalated issues with detailed solutions."
    )
)

# -----------------------------
# Customer Agent (Human proxy)
# -----------------------------
customer_agent = autogen.UserProxyAgent(
    name="customer",
    human_input_mode="ALWAYS",
    code_execution_config={"use_docker": False}
)

# -----------------------------
# Orchestrator Logic
# -----------------------------
def handle_query(query):
    print("\n--- Customer → Support ---\n")

    # Step 1: Customer talks to support
    support_reply = customer_agent.initiate_chat(
        support_agent,
        message=query
    )

    # Extract last message
    last_msg = support_agent.chat_messages[customer_agent][-1]["content"]
    print("\nSupport Agent:", last_msg)

    # Step 2: Escalation check
    if "ESCALATE" in last_msg.upper():
        print("\n--- Escalating to Senior Agent ---\n")

        escalation_reply = support_agent.initiate_chat(
            escalation_agent,
            message=query
        )

        final_msg = escalation_agent.chat_messages[support_agent][-1]["content"]
        print("\nEscalation Agent:", final_msg)


# -----------------------------
# Run the system
# -----------------------------
if __name__ == "__main__":
    try:
        user_query = input("Enter your issue: ")
        handle_query(user_query)

    except Exception as e:
        print("Error:", e)
