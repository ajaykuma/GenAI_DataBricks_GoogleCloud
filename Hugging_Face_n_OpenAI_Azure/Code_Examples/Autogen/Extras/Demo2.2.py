#----------------Refer output of Demo2.1.py--------------------
'''
Multi-Agent Support Chatbot
Customer → Support → Escalation (AutoGen System)

Enter your issue:

##4rth Time (input from user) (response at the end)
> no balance in my savings account

#1st time customer input and response
Conversation
** Customer:** issue with desktop not turning bright

** Support:** If your desktop computer (or laptop) screen is not becoming bright enough, here are some common causes and troubleshooting steps:

1. Check Brightness Settings
Windows:

Go to Settings > System > Display.
Adjust the Brightness slider.
On laptops, you can also use keyboard shortcuts (often keys like F2/F3 or a sun symbol).
Mac:

Use the F1/F2 keys (usually with sun icons) to adjust brightness.
Go to System Preferences > Displays and adjust brightness.
2. Power Plan Settings
On Windows, power-saving modes can dim your screen:
Go to Control Panel > Power Options.
Switch to Balanced or High Performance.
Check advanced power settings for display brightness.
3. Graphics Card Settings
Sometimes, GPU settings override system brightness.
Open Nvidia Control Panel or Intel Graphics Settings.
Look for "Display" or "Color" adjustments.
Reset to defaults or increase brightness.
4. Physical Monitor Settings
If an external monitor is used:
Use the monitor’s buttons to access the On-Screen Display (OSD) menus.
Adjust brightness/contrast from the monitor itself.
5. Software Issues
Recent updates or display drivers can cause brightness problems.
Update Graphics Drivers: Go to Device Manager > Display Adapters > Right-click and update driver.
Roll back drivers if the issue started after an update.
Restart the PC.
6. Ambient Light Sensor (Laptops/tablets)
Some devices auto-adjust brightness using sensors.
Windows: Settings > System > Display > Turn off "Change brightness automatically when lighting changes".
7. Hardware Issues
If none of the above work, possible hardware fault:
Laptop: The backlight (LED or inverter) may be failing.
Desktop Monitor: The monitor itself may be malfunctioning.
Test with a different monitor or device to confirm.
If you need more help:
Specify your device model, operating system, and whether the screen is always dim or only in certain conditions.
Let me know if this is a monitor or laptop screen issue.
Try the steps above and let me know if you need further assistance!

#2nd time customer input and response
** Customer:** issue is fixed now
** Support:** Great to hear the issue is fixed! If you need any more help, just let me know.

#3rd time customer input and response
** Customer:** issue is recurring again

** Support:** I'm sorry to hear that the issue is recurring. Can you provide more details about the problem you're experiencing? We'll try to help you resolve it.

To assist you better, could you please clarify:

What is the issue (error message, behavior)?
When does it occur (after an action, during startup, etc.)?
What troubleshooting steps have you already tried?
Once you give more information, I can offer targeted advice or solutions!

#4rth time customer input and response (shouldnt have ideally responded )
** Customer:** no balance in my savings account

** Support:** If you have no balance in your savings account, here are a few steps you can take:

1. Verify the Account
Double-check your account using your bank’s official website/app to ensure there’s really no balance.
Make sure you’re checking the correct account.
2. Check for Recent Transactions
Look for any withdrawals, automatic payments, or fees that may have emptied your account.
Review your account statement for unexplained activity.
3. Contact Your Bank
If you notice suspicious activity or think your account was compromised, contact your bank immediately.
Ask your bank for a detailed explanation on why your balance is zero.
4. Avoid Overdraft Fees
If your account is linked to checking and goes negative due to withdrawals, be aware of possible overdraft fees.
5. Plan Next Steps
Consider transferring money from another account if possible.
If funds are low, look for ways to increase savings or reduce expenses.
If you share the bank name or specific issue (e.g. unauthorized withdrawal, technical error), I can give more tailored advice.
'''
#----------------Output ends---------------------
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

'''
Additional improvements that can be done
For example:
#1-Issue linking could be smarter
screen looks bad again > screen looks bad again, but it already knew previous issue = flickering / turning off
So better behaviour could be 
Earlier you mentioned the screen was turning off and on.
Is it the same issue or something different like flickering or distortion?

i.e. our code could do > context reinforcement.

#2- Memory is reactive, not structured
Right now:

It remembers conversation
BUT doesn’t store issue states

Ideal system should track:
current_issue = "screen flickering"
status = "resolved" / "recurring"

#3-No explicit issue lifecycle tracking
issue fixed → issue recurring > Bot should have said: It seems the same issue has returned.
Currently it treats it like a new vague issue

--check Demo2.3.py

'''
