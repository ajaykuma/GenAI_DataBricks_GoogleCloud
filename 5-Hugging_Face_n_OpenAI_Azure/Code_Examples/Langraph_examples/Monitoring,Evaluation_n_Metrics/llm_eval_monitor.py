"""
llm_eval_monitor.py
--------------------
A monitoring/evaluation layer for the LangGraph pipelines in
D-2.0.2 / D-2.0.3 / D-2.1 / D-2.2.

Covers the evaluation concepts from LLM_evaluation_n_metrics.txt that are not implemented in 
the original scripts :

  - latency / cost tracking            (LLM system evaluation)
  - token usage accounting             (LLM system evaluation)
  - reference-free heuristic checks    (toxicity flag, refusal flag)
  - reference-free relevancy           (keyword-overlap "answer relevancy")
  - reference-based metrics            (lightweight BLEU-ish / ROUGE-L)
  - human-in-the-loop hooks            (flag_for_review)
  - a golden-dataset-style regression runner

Nothing here requires extra pip installs; everything is stdlib + optional
pandas for the summary table. Swap the heuristic functions for real
libraries (e.g. `detoxify`, `rouge-score`, `bert-score`) when you have
network/package access and want higher-fidelity scores.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ======================================================================
# 1. Data model for a single evaluated LLM call
# ======================================================================

@dataclass
class LLMCallRecord:
    call_id: str
    node_name: str
    product_name: Optional[str]
    prompt: str
    response: str
    latency_sec: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    toxicity_flag: bool
    refusal_flag: bool
    relevancy_score: float          # 0-1, keyword-overlap heuristic
    reference_bleu: Optional[float] = None
    reference_rouge_l: Optional[float] = None
    judge_scores: Optional[Dict[str, float]] = None
    flagged_for_review: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ======================================================================
# 2. Cheap, dependency-free heuristic metrics
# ======================================================================

_TOXIC_KEYWORDS = {
    # illustrative only — replace with a real classifier (e.g. Detoxify,
    # Perspective API, or an LLM-as-judge toxicity prompt) for production use
    "idiot", "stupid", "hate you", "shut up", "kill", "worthless",
}

_REFUSAL_PATTERNS = [
    r"\bI (can(no|')?t|won'?t|am unable to)\b",
    r"\bI'?m sorry, but\b",
    r"\bas an ai\b",
]


def flag_toxicity(text: str) -> bool:
    """Very crude keyword-based toxicity flag. Real deployments should call
    a trained classifier or an LLM-as-judge toxicity check instead."""
    lowered = text.lower()
    return any(kw in lowered for kw in _TOXIC_KEYWORDS)


def flag_refusal(text: str) -> bool:
    """Detects boilerplate refusals / non-answers, useful for spotting
    prompt regressions or over-triggered safety behavior."""
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _REFUSAL_PATTERNS)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_overlap_relevancy(prompt: str, response: str) -> float:
    """Reference-free 'answer relevancy' proxy: what fraction of the
    prompt's distinctive (non-stopword) tokens reappear in the response.
    Cheap stand-in for embedding-similarity relevancy (e.g. BERTScore)."""
    stopwords = {
        "the", "a", "an", "of", "for", "and", "to", "in", "on", "is",
        "this", "that", "with", "product", "brief", "using", "following",
    }
    prompt_tokens = {t for t in _tokenize(prompt) if t not in stopwords and len(t) > 2}
    if not prompt_tokens:
        return 1.0
    response_tokens = set(_tokenize(response))
    overlap = prompt_tokens & response_tokens
    return round(len(overlap) / len(prompt_tokens), 3)


def ngram_overlap_bleu(candidate: str, reference: str, n: int = 2) -> float:
    """Lightweight BLEU-style n-gram precision (no brevity penalty, no
    external deps). Use `sacrebleu`/`nltk` for a rigorous score."""
    def ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    cand_tokens, ref_tokens = _tokenize(candidate), _tokenize(reference)
    if len(cand_tokens) < n or len(ref_tokens) < n:
        return 0.0
    cand_ngrams, ref_ngrams = ngrams(cand_tokens, n), ngrams(ref_tokens, n)
    overlap = sum((cand_ngrams & ref_ngrams).values())
    total = sum(cand_ngrams.values())
    return round(overlap / total, 3) if total else 0.0


def lcs_rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L (F1 over the longest common subsequence). No external deps."""
    c, r = _tokenize(candidate), _tokenize(reference)
    if not c or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(c) + 1)]
    for i in range(1, len(c) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if c[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(c)][len(r)]
    precision, recall = lcs / len(c), lcs / len(r)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


# ======================================================================
# 3. Optional LLM-as-Judge scoring (uses the same llm client you already have)
# ======================================================================

JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge. Score the RESPONSE to the PROMPT
on each criterion from 1 (poor) to 5 (excellent). Base your scores only on what is written;
do not reward length. Return ONLY valid JSON, no other text, with this exact shape:

{{"accuracy": <1-5>, "completeness": <1-5>, "tone_and_style": <1-5>, "cultural_awareness": <1-5>, "rationale": "<one sentence>"}}

PROMPT:
{prompt}

RESPONSE:
{response}
"""


def llm_judge_score(invoke_fn: Callable[[str], str], prompt: str, response: str) -> Dict[str, Any]:
    """
    Generic LLM-as-Judge scorer, matching the criteria the reference doc calls
    out (accuracy, completeness, tone/style, cultural awareness — see the
    "LLM model evaluation metrics" section of LLM_evaluation_n_metrics.txt).

    `invoke_fn` should be a plain `str -> str` function, e.g.:
        invoke_fn = lambda p: llm.invoke([HumanMessage(content=p)]).content
    so this module stays decoupled from any particular LangChain client version.
    """
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)
    raw = invoke_fn(judge_prompt)
    cleaned = raw.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "judge_response_not_json", "raw": raw}


# ======================================================================
# 4. The monitor: wraps a call, scores it, logs it
# ======================================================================

class EvalMonitor:
    """
    Usage:
        monitor = EvalMonitor(log_path="eval_log.jsonl")

        record = monitor.record_call(
            node_name="add_features_benefits",
            product_name=state["product_name"],
            prompt=prompt,
            response=result["message"].content,
            latency_sec=elapsed,
            usage=result["usage"],          # dict with prompt/completion/total tokens, or None
            reference_text=None,            # pass a golden answer if you have one
        )
    """

    def __init__(self, log_path: str = "eval_log.jsonl", flag_relevancy_below: float = 0.15):
        self.log_path = Path(log_path)
        self.flag_relevancy_below = flag_relevancy_below
        self.records: List[LLMCallRecord] = []

    def record_call(
        self,
        node_name: str,
        prompt: str,
        response: str,
        latency_sec: float,
        product_name: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        reference_text: Optional[str] = None,
        judge_invoke_fn: Optional[Callable[[str], str]] = None,
    ) -> LLMCallRecord:
        usage = usage or {}
        relevancy = keyword_overlap_relevancy(prompt, response)

        record = LLMCallRecord(
            call_id=str(uuid.uuid4())[:8],
            node_name=node_name,
            product_name=product_name,
            prompt=prompt,
            response=response,
            latency_sec=round(latency_sec, 3),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            toxicity_flag=flag_toxicity(response),
            refusal_flag=flag_refusal(response),
            relevancy_score=relevancy,
            flagged_for_review=relevancy < self.flag_relevancy_below,
        )

        if reference_text:
            record.reference_bleu = ngram_overlap_bleu(response, reference_text)
            record.reference_rouge_l = lcs_rouge_l(response, reference_text)

        if judge_invoke_fn is not None:
            record.judge_scores = llm_judge_score(judge_invoke_fn, prompt, response)

        self.records.append(record)
        self._append_to_log(record)
        return record

    def _append_to_log(self, record: LLMCallRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    # ------------------------------------------------------------------
    # Aggregation / reporting
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {"n_calls": 0}
        n = len(self.records)
        total_tokens = sum(r.total_tokens or 0 for r in self.records)
        avg_latency = sum(r.latency_sec for r in self.records) / n
        avg_relevancy = sum(r.relevancy_score for r in self.records) / n
        n_flagged = sum(r.flagged_for_review for r in self.records)
        n_toxic = sum(r.toxicity_flag for r in self.records)
        n_refusals = sum(r.refusal_flag for r in self.records)
        return {
            "n_calls": n,
            "total_tokens": total_tokens,
            "avg_latency_sec": round(avg_latency, 3),
            "avg_relevancy": round(avg_relevancy, 3),
            "flagged_for_human_review": n_flagged,
            "toxicity_flags": n_toxic,
            "refusal_flags": n_refusals,
        }

    def to_dataframe(self):
        """Optional: requires pandas. Handy for a Streamlit table/chart."""
        import pandas as pd  # local import so the module works without pandas
        return pd.DataFrame([r.to_dict() for r in self.records])


# ======================================================================
# 5. Golden-dataset style regression runner
# ======================================================================

def run_golden_dataset(
    golden_cases: List[Dict[str, str]],
    generate_fn: Callable[[str], str],
    monitor: EvalMonitor,
    node_name: str = "golden_regression",
) -> Dict[str, Any]:
    """
    golden_cases: [{"input": "...", "expected": "..."}, ...]
    generate_fn:  str -> str, e.g. lambda product_name: run pipeline and
                  return the field you want to check (e.g. final_description)

    Ties together the doc's "Benchmark Datasets" + "Human evaluation" ideas:
    run every case, score it against the reference with BLEU/ROUGE, and
    surface anything below threshold for human review.
    """
    results = []
    for case in golden_cases:
        start = time.time()
        output = generate_fn(case["input"])
        elapsed = time.time() - start
        record = monitor.record_call(
            node_name=node_name,
            prompt=case["input"],
            response=output,
            latency_sec=elapsed,
            reference_text=case.get("expected"),
        )
        results.append(record.to_dict())
    return {"summary": monitor.summary(), "cases": results}
