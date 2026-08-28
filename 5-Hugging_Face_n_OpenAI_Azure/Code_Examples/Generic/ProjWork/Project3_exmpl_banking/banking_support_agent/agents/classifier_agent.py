"""
Classifier Agent

Input: unstructured user message.
Output: one of "Positive Feedback" | "Negative Feedback" | "Query".

This is the entry point of the pipeline; the orchestrator routes on
whatever label this agent returns.
"""

import json
from crewai import Agent, Task, Crew, Process

from agents.llm import get_llm
from config import LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_QUERY, VALID_LABELS

_classifier_agent = Agent(
    role="Banking Support Message Classifier",
    goal=(
        "Read an incoming banking customer support message and classify it "
        "precisely as one of: 'Positive Feedback', 'Negative Feedback', or 'Query'."
    ),
    backstory=(
        "You are a senior triage specialist at a digital bank's customer support "
        "desk. You have read thousands of customer messages and can instantly "
        "tell whether someone is thanking the bank, complaining about an unresolved "
        "problem, or asking a factual question (such as a ticket status). You are "
        "terse, accurate, and never add commentary."
    ),
    llm=get_llm(),
    verbose=False,
    allow_delegation=False,
)


def _build_task(message: str) -> Task:
    return Task(
        description=(
            "Classify the following customer message into exactly one category: "
            f"{', '.join(VALID_LABELS)}.\n\n"
            f'Customer message: "{message}"\n\n'
            "Rules:\n"
            "- 'Positive Feedback': the customer is thanking the bank, praising service, "
            "or expressing satisfaction about something already resolved.\n"
            "- 'Negative Feedback': the customer is complaining about an unresolved or "
            "ongoing problem, expressing frustration, or reporting something that went wrong "
            "and has NOT been fixed. This does NOT include messages that only ask about a "
            "ticket status.\n"
            "- 'Query': the customer is asking a question, most commonly requesting the "
            "status of an existing ticket (may include a ticket number).\n\n"
            'Respond with ONLY a JSON object of the form {"label": "<one of the three categories>"}. '
            "No other text."
        ),
        expected_output='A JSON object like {"label": "Query"}',
        agent=_classifier_agent,
    )


def classify_message(message: str) -> str:
    """
    Run the Classifier Agent on a single message and return a normalized
    label from config.VALID_LABELS. Falls back to 'Query' if the model
    output can't be parsed, so the pipeline never crashes on a bad label.
    """
    task = _build_task(message)
    crew = Crew(agents=[_classifier_agent], tasks=[task], process=Process.sequential, verbose=False)
    raw_result = str(crew.kickoff())

    label = _parse_label(raw_result)
    return label


def _parse_label(raw_result: str) -> str:
    text = raw_result.strip()
    # Strip markdown code fences if the model wrapped its JSON in them.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
        label = data.get("label", "").strip()
        if label in VALID_LABELS:
            return label
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: substring match against the raw text.
    for candidate in VALID_LABELS:
        if candidate.lower() in text.lower():
            return candidate

    return LABEL_QUERY  # safest default: route to a human-readable "couldn't classify" query flow
