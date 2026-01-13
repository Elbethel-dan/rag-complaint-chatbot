# =============================================================================
# evaluation.py - Complaint RAG Evaluation Helpers
# =============================================================================
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd

from .rag_pipeline import RAGPipeline

EVALUATION_QUESTIONS = [
    # Credit Card
    "Why do customers report being charged unexpected fees on their credit cards?",
    "What are the common complaints regarding credit card customer service or dispute handling?",
    # Personal Loan
    "Why are some personal loan applications denied, according to customer complaints?",
    "What issues do customers report about personal loan repayment or interest rates?",
    # Savings Account
    "What problems do customers report with opening or managing savings accounts?",
    "Are there complaints about delayed transactions or account access issues in savings accounts?",
    # Money Transfers
    "What are the most frequent complaints customers have about domestic or international money transfers?",
    "Why do customers report delays or failures when making money transfers?"
]


@dataclass
class EvaluationResult:
    question: str
    answer: str
    sources_summary: str
    score: Optional[int] = None
    comments: str = ""


@dataclass
class EvaluationReport:
    results: List[EvaluationResult] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def average_score(self) -> float:
        scored = [r.score for r in self.results if r.score is not None]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def num_evaluated(self) -> int:
        return sum(1 for r in self.results if r.score is not None)


def summarize_sources(sources: list, max_sources: int = 2) -> str:
    if not sources:
        return "No sources retrieved"
    summaries = []
    for doc in sources[:max_sources]:
        c_id = doc['metadata'].get('complaint_id', 'N/A')
        product = doc['metadata'].get('product', 'N/A')
        text_preview = doc.get('text', '')[:100].replace('\n', ' ')
        summaries.append(f"[{c_id}] {product}: {text_preview}...")
    return " | ".join(summaries)


def run_evaluation(
    rag_pipeline: RAGPipeline,
    questions: List[str],
    k: int = 5,
    verbose: bool = True
) -> EvaluationReport:
    report = EvaluationReport(
        settings={"num_questions": len(questions), "retrieval_k": k}
    )

    for i, question in enumerate(questions, 1):
        if verbose:
            print(f"\n[{i}/{len(questions)}] {question[:50]}...")

        # Retrieve chunks
        retrieved_chunks = rag_pipeline.retriever.retrieve(question, k=k)
        answer = rag_pipeline.generator.generate(question, retrieved_chunks)

        # Create evaluation result
        result = EvaluationResult(
            question=question,
            answer=answer,
            sources_summary=summarize_sources(retrieved_chunks)
        )

        report.results.append(result)

        if verbose:
            print(f"    Answer: {answer[:100]}...")

    return report


def results_to_dataframe(report: EvaluationReport) -> pd.DataFrame:
    data = []
    for r in report.results:
        data.append({
            "Question": r.question,
            "Generated Answer": r.answer,
            "Retrieved Sources": r.sources_summary,
            "Quality Score": r.score,
            "Comments": r.comments
        })
    return pd.DataFrame(data)


def results_to_markdown_table(report: EvaluationReport) -> str:
    lines = [
        "| # | Question | Generated Answer | Sources | Score | Comments |",
        "|---|----------|----------------|--------|-------|---------|",
    ]
    for i, r in enumerate(report.results, 1):
        q = (r.question[:40] + "...") if len(r.question) > 40 else r.question
        a = (r.answer[:60] + "...") if len(r.answer) > 60 else r.answer
        s = (r.sources_summary[:40] + "...") if len(r.sources_summary) > 40 else r.sources_summary
        sc = str(r.score) if r.score else "-"
        c = (r.comments[:30] + "...") if len(r.comments) > 30 else r.comments
        # Escape pipes
        q, a, s, c = map(lambda x: x.replace("|", "\\|"), [q, a, s, c])
        lines.append(f"| {i} | {q} | {a} | {s} | {sc} | {c} |")
    lines.append("")
    lines.append(f"**Average Score:** {report.average_score:.2f} / 5.0")
    lines.append(f"**Questions Evaluated:** {report.num_evaluated} / {len(report.results)}")
    return "\n".join(lines)
