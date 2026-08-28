"""
Orchestrator

Ties the three agents together into the full pipeline described in the
project spec:

    Classifier Agent -> (Feedback Handler Agent | Query Handler Agent)

CrewAI Crews are best suited to fixed/sequential task chains; since this
pipeline branches on the classifier's output, the branching itself is plain
Python (a "manager" layer), while each branch's actual work is delegated to
a CrewAI agent. This keeps routing logic easy to test and debug while still
using CrewAI for every piece of LLM-driven work.
"""

from dataclasses import dataclass
from typing import Optional
import traceback

from agents.classifier_agent import classify_message
from agents.feedback_agent import handle_positive_feedback, handle_negative_feedback
from agents.query_agent import handle_query
from config import LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_QUERY, DEFAULT_CUSTOMER_NAME
import logger


@dataclass
class PipelineResult:
    user_input: str
    classification: str
    agent_path: str
    response: str
    ticket_number: Optional[str]
    success: bool
    error: Optional[str] = None


def run_pipeline(user_input: str, customer_name: str = DEFAULT_CUSTOMER_NAME) -> PipelineResult:
    """
    Run the full multi-agent pipeline on a single user message:
    classify -> route -> handle -> log.
    """
    if not user_input or not user_input.strip():
        result = PipelineResult(
            user_input=user_input,
            classification="N/A",
            agent_path="None",
            response="Please enter a message.",
            ticket_number=None,
            success=False,
            error="Empty input",
        )
        _log(result)
        return result

    try:
        classification = classify_message(user_input)
    except Exception as exc:  # noqa: BLE001 - surface any LLM/config error to the UI
        traceback.print_exc()
        result = PipelineResult(
            user_input=user_input,
            classification="Error",
            agent_path="Classifier Agent",
            response="Sorry, something went wrong while classifying your message.",
            ticket_number=None,
            success=False,
            error=str(exc),
        )
        _log(result)
        return result

    try:
        if classification == LABEL_POSITIVE:
            outcome = handle_positive_feedback(user_input, customer_name)
            agent_path = "Classifier Agent -> Feedback Handler Agent (Positive)"
        elif classification == LABEL_NEGATIVE:
            outcome = handle_negative_feedback(user_input, customer_name)
            agent_path = "Classifier Agent -> Feedback Handler Agent (Negative)"
        else:  # LABEL_QUERY
            outcome = handle_query(user_input)
            agent_path = "Classifier Agent -> Query Handler Agent"

        result = PipelineResult(
            user_input=user_input,
            classification=classification,
            agent_path=agent_path,
            response=outcome["response"],
            ticket_number=outcome.get("ticket_number"),
            success=True,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        result = PipelineResult(
            user_input=user_input,
            classification=classification,
            agent_path="Handler Agent",
            response="Sorry, something went wrong while handling your message.",
            ticket_number=None,
            success=False,
            error=str(exc),
        )

    _log(result)
    return result


def _log(result: PipelineResult) -> None:
    logger.log_interaction(
        user_input=result.user_input,
        classification=result.classification,
        agent_path=result.agent_path,
        response=result.response,
        ticket_number=result.ticket_number,
        success=result.success,
        error=result.error,
    )
