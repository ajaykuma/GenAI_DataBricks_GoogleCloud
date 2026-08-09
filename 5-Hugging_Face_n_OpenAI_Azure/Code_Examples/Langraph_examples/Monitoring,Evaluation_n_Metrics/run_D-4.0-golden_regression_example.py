"""
run_golden_regression_example.py
---------------------------------
A non-Streamlit, CI-friendly example of the "Benchmark Datasets" +
"Human evaluation" ideas from LLM_evaluation_n_metrics.txt: run a fixed
set of golden (input, expected-output) pairs through your pipeline,
score each one, and fail/flag anything that regresses.

Run it directly:  python run_golden_regression_example.py
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage

from llm_eval_monitor import EvalMonitor, run_golden_dataset

load_dotenv("E:\\Lesson_2_demos\\.env")

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    deployment_name="gpt-4.1",
    temperature=0.2,  # low temperature for reproducible regression checks
)


def generate_basic_description(product_name: str) -> str:
    """Minimal stand-in for the 'basic_description' node from the D-2.x
    scripts, isolated so it can be regression-tested on its own."""
    prompt = f"Write a brief description of a product called '{product_name}'."
    return llm.invoke([HumanMessage(content=prompt)]).content


# A tiny golden dataset. In practice this would live in a CSV/JSON file
# reviewed by a human (the "golden dataset" + "human in the loop" ideas
# from the reference doc) and grow over time as regressions are found.
GOLDEN_CASES = [
    {
        "input": "AquaTrack Smart Bottle",
        "expected": "A smart water bottle that tracks your hydration and reminds you to drink water throughout the day.",
    },
    {
        "input": "EcoCharge Solar Power Bank",
        "expected": "A portable solar-powered charger that lets you recharge your devices anywhere using sunlight.",
    },
    {
        "input": "PetPal GPS Collar",
        "expected": "A GPS-enabled pet collar that lets owners track their pet's location in real time from a phone app.",
    },
]


def main():
    monitor = EvalMonitor(log_path="golden_regression_log.jsonl")
    report = run_golden_dataset(
        golden_cases=GOLDEN_CASES,
        generate_fn=generate_basic_description,
        monitor=monitor,
        node_name="generate_basic_description",
    )

    print("=== Summary ===")
    for k, v in report["summary"].items():
        print(f"{k}: {v}")

    print("\n=== Per-case scores ===")
    for case in report["cases"]:
        print(
            f"[{case['call_id']}] rouge_l={case['reference_rouge_l']} "
            f"bleu={case['reference_bleu']} relevancy={case['relevancy_score']} "
            f"flagged={case['flagged_for_review']}"
        )
        if case["flagged_for_review"]:
            print(f"   -> REVIEW response: {case['response'][:200]}")

    # Simple pass/fail gate you could wire into CI: fail the build if
    # average ROUGE-L against golden answers drops below a threshold.
    rouge_scores = [c["reference_rouge_l"] for c in report["cases"] if c["reference_rouge_l"] is not None]
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    threshold = 0.15
    print(f"\nAverage ROUGE-L: {avg_rouge:.3f} (threshold: {threshold})")
    if avg_rouge < threshold:
        raise SystemExit("Golden regression FAILED: quality dropped below threshold.")
    print("Golden regression PASSED.")


if __name__ == "__main__":
    main()
