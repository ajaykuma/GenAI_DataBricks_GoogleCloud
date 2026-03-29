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

'''
Sample output and issues fixed

Multi-Agent IT Support Chatbot
Context-aware + Structured Memory + Domain Control

Enter your issue:

all issues resolved

Conversation
Customer: what is volcano

Support: I can only assist with IT support-related issues (computers, software, networks). Please ask a relevant question.

Customer: what is volcano in IT

Support: You’ve asked multiple times about "volcano" in IT. Here’s a clear, structured answer based on IT context:

Volcano in IT:

There is no widely recognized IT term "volcano." It is not a standard concept in computers, software, or networking.
If you are referring to a specific software, hardware, or project named "Volcano," please provide more details (e.g., "Volcano Tool" for mobile phone flashing and repair).
Troubleshooting Steps:

Please clarify the context or the product you mean by "volcano" (e.g., mobile repair tool, software, hardware).
If you have an issue with a tool or software named "Volcano," describe the problem (error messages, steps you’ve taken, etc.).
Issue Status:
Active — awaiting clarification on your IT-related question about "volcano."

Let me know more details so I can assist you effectively.

Customer: having issue with desktop screen which turns off and on

Support: Thank you for reporting the desktop screen issue (screen turns off and on). This is now the active issue.

Troubleshooting Steps
Check Cable Connections

Ensure both the power cable and display cable (HDMI/VGA/DisplayPort) are securely connected to both the monitor and the desktop.
Test the Monitor

If possible, connect your monitor to a different computer or laptop.
Does the issue persist?
Yes: The problem is likely with the monitor or its cables.
No: The issue may be with your desktop.
Check Power Supply

Verify the power outlet and power strip are working properly. Try plugging the monitor directly into a wall socket.
Replace Cables

Swap out the display cable and power cable with known working ones.
Update Graphics Drivers

If the monitor works on other devices, boot your desktop in Safe Mode.
Update or reinstall your graphics drivers:
Windows: Device Manager → Display adapters → Right-click your GPU → Update driver.
Check for Overheating

Ensure your desktop is not overheating. Overheating can cause the display to turn off and on.
Monitor Settings

Check the monitor’s built-in menu for power-saving or auto-off features.
Next Steps
Please let me know at which step you encounter an issue or if any of these actions resolve the problem.
If the issue is recurring after troubleshooting, I’ll track it and escalate as needed.
Issue status: Active — awaiting your feedback on the troubleshooting steps above.

Customer: issue was related to software upgrade, i downgraded it and not it works fine but now i have issue with keys on keyboard

Support: Thank you for the update. The desktop screen issue has been resolved after downgrading the software.
Now, you have a new active issue: keys on your keyboard are not working correctly.

Troubleshooting Steps: Keyboard Issue
Check Physical Connection

If you are using a wired keyboard, unplug it and plug it back in.
For wireless keyboards, check the battery and reconnect.
Test Keyboard on Another Device

Connect your keyboard to a different computer.
If it works, the issue is likely with your desktop settings.
If it does not work, your keyboard may be faulty.
Check for Driver Issues

Go to Device Manager → Keyboards.
Right-click your keyboard → Update driver.
If necessary, uninstall the device and restart your computer to reinstall drivers automatically.
Check Language and Input Settings

Go to Settings → Time & Language → Language → Keyboard.
Ensure the correct layout is selected.
Test in Safe Mode

Restart your computer in Safe Mode and see if the keyboard works.
If it works in Safe Mode, a third-party application or driver may be causing the issue.
Check for Software Conflicts

Since the issue appeared after a software downgrade, review any remaining software or drivers that may have changed.
Uninstall any keyboard-related utilities or third-party software.
Try Another Keyboard

If possible, use a different keyboard to see if the issue persists.
Next Steps
Please let me know which keys are not working (all keys, specific keys, or random keys).
Let me know the results after each troubleshooting step.
If this keyboard issue becomes recurring or persists after troubleshooting, I will track it accordingly.
Issue status:
Active — keyboard issue (awaiting your feedback).
Resolved — desktop screen issue (after software downgrade).

Customer: keyboard issue is fixed but desktop issue is back

Support: Thank you for your update. I acknowledge that your desktop screen issue (screen turns off and on) is recurring. Your keyboard issue remains resolved.

Recurring Desktop Screen Issue — Troubleshooting Steps
Since the issue returned after a software downgrade, let's focus on both hardware and software causes:

1. Confirm Hardware Connections (Repeat)
Secure all cables: Double-check that the power and display cables are firmly connected.
Try different cables/ports: Swap to known working cables and try different ports on both monitor and desktop.
2. Test Monitor and Desktop Separately
Monitor test: Connect your monitor to another device. Does the issue occur?
Yes: Monitor or cable issue.
No: Likely desktop-related.
Alternate monitor: Connect a different monitor to your desktop. Does the issue persist?
3. Power Source Check
Plug the monitor and desktop directly into a wall outlet (avoid power strips/surge protectors).
4. Graphics Driver Reinstallation
Since you downgraded software, the graphics driver may be mismatched or corrupted.
Go to Device Manager → Display adapters.
Right-click your GPU → Uninstall device.
Restart your PC. Windows should reinstall the correct drivers automatically.
Alternatively, download the latest driver from your GPU manufacturer’s website.
5. Check for Overheating
Ensure all fans are working and vents are clear.
Use a hardware monitoring tool (like HWMonitor) to check CPU/GPU temperatures.
6. Review Power Settings
Go to Control Panel → Power Options.
Ensure sleep/hibernate/turn off display settings are not set too aggressively.
7. Event Viewer Check
Open Event Viewer (Windows key, type "Event Viewer").
Under Windows Logs → System, look for critical errors or warnings at the times the screen turns off.
Next Steps
Please report which steps you complete and any findings (especially if the issue happens with another monitor or device).
If the issue persists after all steps, let me know so I can suggest advanced diagnostics or escalation.
Issue status:

Active (recurring) — Desktop screen turns off and on
Resolved — Keyboard issue
Let me know your results for further guidance.

Customer: all issues resolved

Support: Thank you for confirming.
Status update:

Resolved — Desktop screen issue (recurring): No longer occurring.
Resolved — Keyboard issue: No longer occurring.
If any of these issues return or you encounter a new IT-related problem, please let me know and I will track and assist accordingly.

Issue lifecycle:

Desktop screen issue — resolved (was recurring)
Keyboard issue — resolved
No active IT issues at this time.
If you need further IT support, I’m here to help!
'''
