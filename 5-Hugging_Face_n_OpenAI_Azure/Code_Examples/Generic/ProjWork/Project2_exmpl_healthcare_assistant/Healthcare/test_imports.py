import traceback

print("Testing tools...")
try:
    from agent.tools import get_patient_history
    print("tools OK")
except Exception as e:
    traceback.print_exc()

print("\nTesting memory...")
try:
    from agent.memory import retrieve_context
    print("memory OK")
except Exception as e:
    traceback.print_exc()

print("\nTesting graph...")
try:
    from agent.graph import healthcare_agent
    print("graph OK")
except Exception as e:
    traceback.print_exc()