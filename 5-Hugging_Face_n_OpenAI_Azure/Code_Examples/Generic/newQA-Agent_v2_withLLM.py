'''
<class 'langchain_core.messages.ai.AIMessage'>
```python

Simple QA Test Case Generator Agent using AzureChatOpenAI (GPT-5.1)

Prereqs:
- `pip install openai`
- Azure OpenAI resource with model `gpt-5.1` deployed as a chat model
- Environment variables set:
  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
'''

import os
from openai import AzureOpenAI

# --------- Azure OpenAI Client Setup ---------
client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview",  # check and update to current API version if needed
)

DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT")  # e.g. "gpt-5.1"

# --------- QA Agent Function ---------
def generate_test_cases(requirements: str, code_snippet: str = "") -> str:
    """
    Build a simple QA Agent that takes developer inputs and generates test cases.
    :param requirements: Description of the feature / user story / API behavior.
    :param code_snippet: Optional Python code or function being tested.
    :return: Suggested test cases in structured form.
    """

    system_prompt = """
You are a QA Engineer. 
Given feature requirements and optional code, produce clear, structured test cases.

Requirements for output:
- Group tests by category (e.g., Positive, Negative, Edge, Performance if relevant)
- For each test: include title, preconditions, steps, expected result
- Keep the output concise but complete enough to implement.
    """

    user_prompt = f"""
Feature / requirements:
{requirements}

Optional code:
{code_snippet}

Generate test cases now.
    """

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,  # this should point to your deployed gpt-5.1 Azure Chat model
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# --------- Example Usage (CLI) ---------
if __name__ == "__main__":
    print("=== QA Test Case Generator (Azure OpenAI / GPT-5.1) ===")
    print("Enter feature requirements (end with an empty line):")

    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    requirements_text = "\n".join(lines)

    print("\nOptional: paste relevant code snippet (end with an empty line).")
    print("Press Enter on an empty line if you don't have code.")
    code_lines = []
    while True:
        line = input()
        if not line.strip():
            break
        code_lines.append(line)
    code_text = "\n".join(code_lines)

    print("\nGenerating test cases...\n")
    test_cases = generate_test_cases(requirements_text, code_text)
    print(test_cases)

'''
Notes:
- Replace `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_OPENAI_DEPLOYMENT` with your Azure values (or set them as environment variables).
- `AZURE_OPENAI_DEPLOYMENT` should be the name of your Azure chat deployment that uses `gpt-5.1`.
- The example uses the `AzureOpenAI` client and `chat.completions.create`, which is the pattern for Azure Chat models.'''
