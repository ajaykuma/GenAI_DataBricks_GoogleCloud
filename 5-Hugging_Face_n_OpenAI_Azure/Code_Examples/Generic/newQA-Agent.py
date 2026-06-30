'''<class 'langchain_core.messages.ai.AIMessage'>
Here’s a minimal, self‑contained Python example of a simple QA agent that:

- Takes plain‑text feature descriptions / requirements from developers
- Generates basic test cases (as structured data)
- Prints them out in a readable format

This version does **not** depend on any external AI service, so you can run it directly.

python'''

"""
Simple QA Agent in Python

- Takes developer input describing a feature or function
- Generates basic test cases using simple rule-based logic
"""

from typing import List, Dict

class QAAgent:
    def __init__(self):
        pass

    def generate_test_cases(self, requirement: str) -> List[Dict]:
        """
        Very simple heuristic-based test case generator.
        In a real system, you would replace this with an LLM call or
        more advanced parsing/analysis.
        """
        req_lower = requirement.lower()
        test_cases = []

        # Generic test case template
        def make_test_case(id_suffix, title, steps, expected):
            return {
                "id": f"TC_{id_suffix}",
                "title": title,
                "steps": steps,
                "expected_result": expected
            }

        # 1. Generic smoke test
        test_cases.append(
            make_test_case(
                "SMOKE",
                "Basic functionality smoke test",
                [
                    f"Set up environment according to requirement: '{requirement}'",
                    "Execute the primary scenario described",
                ],
                "Feature works as described without errors"
            )
        )

        # 2. Boundary / invalid input tests if it looks like input handling
        if any(word in req_lower for word in ["input", "field", "form", "request", "api"]):
            test_cases.append(
                make_test_case(
                    "INVALID_INPUT",
                    "Invalid input handling",
                    [
                        "Provide clearly invalid input (e.g., empty, too long, wrong type)",
                        "Observe the system response"
                    ],
                    "System should reject invalid input with a clear error, no crash"
                )
            )
            test_cases.append(
                make_test_case(
                    "BOUNDARY_VALUES",
                    "Boundary value analysis",
                    [
                        "Identify min/max or special boundary values from requirements",
                        "Send those boundary values through the feature",
                    ],
                    "System should handle boundary values correctly per requirements"
                )
            )

        # 3. Auth / permissions tests if mentioned
        if any(word in req_lower for word in ["login", "auth", "authentication", "authorization", "role", "permission"]):
            test_cases.append(
                make_test_case(
                    "AUTH_REQUIRED",
                    "Access control check",
                    [
                        "Attempt to use the feature as an unauthenticated user",
                        "Attempt to use the feature with different roles/permissions"
                    ],
                    "Only authorized users can access; unauthorized users are blocked"
                )
            )

        # 4. Performance / load tests if mentioned
        if any(word in req_lower for word in ["performance", "load", "concurrent", "scale", "scalability"]):
            test_cases.append(
                make_test_case(
                    "LOAD_TEST",
                    "Basic load test",
                    [
                        "Simulate multiple concurrent requests/users",
                        "Measure response times and error rate"
                    ],
                    "System meets performance requirements under expected load"
                )
            )

        # 5. Data consistency / DB tests if mentioned
        if any(word in req_lower for word in ["database", "db", "persist", "save", "storage"]):
            test_cases.append(
                make_test_case(
                    "DATA_PERSISTENCE",
                    "Data persistence and consistency",
                    [
                        "Perform create/update/delete operations",
                        "Verify data is stored correctly",
                        "Check that subsequent reads return consistent data"
                    ],
                    "Data is persisted and remains consistent per requirements"
                )
            )

        # 6. Default “happy path” test
        test_cases.append(
            make_test_case(
                "HAPPY_PATH",
                "Happy path scenario",
                [
                    "Follow the typical usage flow described by the developer",
                    "Use valid, expected inputs",
                ],
                "System behaves exactly as described in the requirement"
            )
        )

        return test_cases


def print_test_cases(test_cases: List[Dict]) -> None:
    """Pretty-print test cases to console."""
    for tc in test_cases:
        print("=" * 60)
        print(f"Test Case ID    : {tc['id']}")
        print(f"Title           : {tc['title']}")
        print("Steps:")
        for i, step in enumerate(tc["steps"], start=1):
            print(f"  {i}. {step}")
        print(f"Expected Result : {tc['expected_result']}")
        print("=" * 60)
        print()


def main():
    agent = QAAgent()
    print("=== Simple QA Agent ===")
    print("Enter a feature/requirement description from the developer.")
    print("Type 'quit' to exit.\n")

    while True:
        requirement = input("Developer requirement> ").strip()
        if requirement.lower() in ("quit", "exit"):
            print("Exiting QA Agent.")
            break

        if not requirement:
            print("Please enter a non-empty requirement.\n")
            continue

        test_cases = agent.generate_test_cases(requirement)
        print("\nGenerated Test Cases:\n")
        print_test_cases(test_cases)


if __name__ == "__main__":
    main()

'''
### How to use

1. Save as `qa_agent.py`.
2. Run:

   ```bash
   python qa_agent.py
   ```

3. When prompted, paste a developer requirement, e.g.:

   > API endpoint to create a new user with email and password. Must validate input and require authentication.

4. The script will print generated test cases (smoke, invalid input, auth, etc.).

---

If you want a version that uses an LLM (e.g., OpenAI, local model)
to generate richer test cases, I can provide a second example that plugs into an API.'''
