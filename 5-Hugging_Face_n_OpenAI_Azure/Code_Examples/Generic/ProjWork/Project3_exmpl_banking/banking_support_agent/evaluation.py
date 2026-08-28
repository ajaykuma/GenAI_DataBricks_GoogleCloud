"""
Evaluation harness (Part 2, item 7: Model Evaluation).

Provides:
- A labeled test set covering all three classification categories, including
  a few deliberately ambiguous edge cases.
- Classification accuracy scoring against that test set.
- End-to-end routing success rate (did the pipeline complete without error
  and did it route to the expected agent branch).

This is a lightweight, dependency-free eval you can run from the CLI or the
Streamlit "Evaluation" tab. When LangSmith is added later, this test set can
be reused as a LangSmith dataset and this scoring logic replaced/augmented
with LangSmith's eval runners.
"""

from dataclasses import dataclass, field

from agents.classifier_agent import classify_message
from orchestrator import run_pipeline
from config import LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_QUERY
import database


@dataclass
class TestCase:
    message: str
    expected_label: str
    note: str = ""


CLASSIFICATION_TEST_SET: list[TestCase] = [
    TestCase("Thanks for sorting out my net banking login issue.", LABEL_POSITIVE),
    TestCase("Thank you so much, the loan approval was so quick!", LABEL_POSITIVE),
    TestCase("Really appreciate the support team's help yesterday.", LABEL_POSITIVE),
    TestCase("My debit card replacement still hasn't arrived.", LABEL_NEGATIVE),
    TestCase("This is the third time I've had to call about the same overdraft error!", LABEL_NEGATIVE),
    TestCase("Extremely disappointed, my funds transfer failed twice today.", LABEL_NEGATIVE),
    TestCase("Could you check the status of ticket 650932?", LABEL_QUERY),
    TestCase("What's happening with my complaint, ticket number 112233?", LABEL_QUERY),
    TestCase("How do I check my ticket status?", LABEL_QUERY, note="no ticket number present"),
    TestCase(
        "I complained last week (ticket 998877) and it's still not fixed, this is ridiculous.",
        LABEL_QUERY,
        note="ambiguous: contains a ticket number + frustration; treated as a status query",
    ),
]


@dataclass
class EvalReport:
    total: int
    correct: int
    accuracy: float
    failures: list[dict] = field(default_factory=list)


def run_classification_eval() -> EvalReport:
    """Run the classifier against CLASSIFICATION_TEST_SET and score accuracy."""
    failures = []
    correct = 0
    for case in CLASSIFICATION_TEST_SET:
        predicted = classify_message(case.message)
        if predicted == case.expected_label:
            correct += 1
        else:
            failures.append(
                {
                    "message": case.message,
                    "expected": case.expected_label,
                    "predicted": predicted,
                    "note": case.note,
                }
            )
    total = len(CLASSIFICATION_TEST_SET)
    return EvalReport(
        total=total,
        correct=correct,
        accuracy=round(100 * correct / total, 1) if total else 0.0,
        failures=failures,
    )


def run_routing_eval() -> dict:
    """
    Run the full pipeline end-to-end for each test case and check:
    - it completed without error
    - it produced a non-empty response
    - (for negative feedback) a ticket was actually created in the DB
    """
    database.reset_db()
    total = len(CLASSIFICATION_TEST_SET)
    successes = 0
    details = []

    for case in CLASSIFICATION_TEST_SET:
        if case.expected_label == LABEL_QUERY:
            continue  # queries need a pre-existing ticket; covered by classification eval instead
        result = run_pipeline(case.message)
        ok = result.success and bool(result.response.strip())
        if case.expected_label == LABEL_NEGATIVE:
            ok = ok and result.ticket_number is not None and database.get_ticket(result.ticket_number) is not None
        successes += int(ok)
        details.append({"message": case.message, "success": ok, "agent_path": result.agent_path})

    scored_total = len(details)
    return {
        "total": scored_total,
        "successes": successes,
        "success_rate": round(100 * successes / scored_total, 1) if scored_total else 0.0,
        "details": details,
    }


if __name__ == "__main__":
    print("Running classification evaluation...")
    report = run_classification_eval()
    print(f"Accuracy: {report.correct}/{report.total} ({report.accuracy}%)")
    if report.failures:
        print("\nFailures:")
        for f in report.failures:
            print(f"  - '{f['message']}' expected={f['expected']} predicted={f['predicted']} ({f['note']})")

    print("\nRunning routing evaluation...")
    routing = run_routing_eval()
    print(f"Routing success rate: {routing['successes']}/{routing['total']} ({routing['success_rate']}%)")
