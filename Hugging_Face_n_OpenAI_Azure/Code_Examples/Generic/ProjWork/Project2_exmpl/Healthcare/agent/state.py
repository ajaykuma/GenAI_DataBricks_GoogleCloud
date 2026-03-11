# Agent state schema
#LangGraph passes this state between nodes. 
# Each node reads from and writes to it, enabling clean task chaining.

from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    patient_id: Optional[str]
    patient_context: Optional[str]
    task_plan: Optional[List[str]]
    retrieved_history: Optional[str]
    appointment_result: Optional[str]
    search_result: Optional[str]
    final_response: Optional[str]
