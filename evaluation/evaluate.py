"""Task 16 - RAGAS evaluation of the RAG chatbot."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FaithfulnesswithHHEM,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from backend.llm_factory import get_llm


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "evaluation" / "ragas_input.json"
RESULTS_PATH = BASE_DIR / "evaluation" / "ragas_results.json"
MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"


def load_input() -> dict:
    """Load the already-generated RAG evaluation data."""

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict) -> None:
    """Save results as valid JSON."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def calculate_scores(dataframe, column_name: str) -> dict:
    """Calculate valid scores and their average."""

    raw_values = dataframe[column_name].tolist()

    cleaned_scores = []
    valid_scores = []

    for value in raw_values:

        try:
            if value is None:
                cleaned_scores.append(None)
                continue

            score = float(value)

            if score != score:
                cleaned_scores.append(None)
                continue

            cleaned_scores.append(score)
            valid_scores.append(score)

        except (TypeError, ValueError):
            cleaned_scores.append(None)

    average = (
        sum(valid_scores) / len(valid_scores)
        if valid_scores
        else None
    )

    return {
        "scores": cleaned_scores,
        "valid_scores": valid_scores,
        "average": average,
        "evaluated_samples": len(valid_scores),
        "total_samples": len(raw_values),
    }


def evaluate_metric(
    dataset: Dataset,
    metric,
    metric_name: str,
    column_name: str,
    evaluator_llm=None,
    evaluator_embeddings=None,
):
    """Run one RAGAS metric."""

    print("\n==============================================")
    print(f"EVALUATING: {metric_name}")
    print("==============================================")

    result = evaluate(
        dataset=dataset,
        metrics=[metric],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        batch_size=1,
        run_config=RunConfig(
            timeout=90,
            max_retries=0,
            max_workers=1,
        ),
        raise_exceptions=False,
        show_progress=True,
    )

    dataframe = result.to_pandas()

    print("\nRAGAS result columns:")
    print(list(dataframe.columns))

    if column_name not in dataframe.columns:
        raise RuntimeError(
            f"Expected column '{column_name}' was not returned by RAGAS."
        )

    metric_result = calculate_scores(
        dataframe,
        column_name,
    )

    print(f"\n{metric_name} completed.")

    if metric_result["average"] is not None:
        print(
            f"Average score: "
            f"{metric_result['average']:.4f}"
        )
    else:
        print("Average score: No valid scores")

    print(
        f"Valid samples: "
        f"{metric_result['evaluated_samples']}/"
        f"{metric_result['total_samples']}"
    )

    return metric_result


def main() -> None:

    print("\n==============================================")
    print("TASK 16 RAGAS EVALUATION")
    print("==============================================")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"RAGAS input file not found: {INPUT_PATH}"
        )
    data = load_input()

    questions = data["question"]
    answers = data["answer"]
    contexts = data["contexts"]
    ground_truths = data["ground_truth"]

    # Load independent reference contexts for RAGAS context metrics.
    reference_dataset_path = BASE_DIR / "evaluation" / "test_dataset.json"

    with open(reference_dataset_path, "r", encoding="utf-8-sig") as file:
        reference_data = json.load(file)

    reference_by_question = {
        item["question"]: item["contexts"]
        for item in reference_data
    }

    reference_contexts = []

    for question in questions:
        if question not in reference_by_question:
            raise ValueError(
                f"Reference context missing for question: {question}"
            )

        reference_contexts.append(
            reference_by_question[question]
        )

    total = len(questions)

    if total != 10:
        raise ValueError(
            f"Task 16 requires exactly 10 questions, but found {total}."
        )

    print(f"Loaded existing RAGAS input: {total} questions.")
    print("Fresh RAG answers will NOT be generated again.")

    dataset = Dataset.from_dict(
        {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference_contexts": reference_contexts,
            "reference": ground_truths,
        }
    )

    # --------------------------------------------------
    # Local evaluator embeddings
    # --------------------------------------------------

    print("\nLoading local evaluator embeddings...")

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=str(MODEL_PATH)
        )
    )

    print("Evaluator embeddings loaded successfully.")

    # --------------------------------------------------
    # Results structure
    # --------------------------------------------------

    final_results = {
        "generated_at": datetime.now().isoformat(),
        "total_questions": total,
        "pipeline_errors": data.get("errors", []),
    }

    # --------------------------------------------------
    # 1. Faithfulness
    #
    # HHEM performs local NLI-based evaluation and avoids
    # using the slow Qwen model as a judge.
    # --------------------------------------------------

    try:

        faithfulness_metric = FaithfulnesswithHHEM(
            device="cpu",
            batch_size=10,
        )

        final_results["Faithfulness"] = evaluate_metric(
            dataset=dataset,
            metric=faithfulness_metric,
            metric_name="Faithfulness",
            column_name="faithfulness_with_hhem",
            evaluator_embeddings=evaluator_embeddings,
        )

        save_json(
            RESULTS_PATH,
            final_results,
        )

    except Exception as error:

        print("\nFaithfulness failed:")
        print(repr(error))

        final_results["Faithfulness"] = {
            "error": repr(error)
        }

        save_json(
            RESULTS_PATH,
            final_results,
        )

    # --------------------------------------------------
    # 2. Context Precision
    #
    # Non-LLM metric: no Qwen evaluation required.
    # --------------------------------------------------

    try:

        context_precision_metric = (
            NonLLMContextPrecisionWithReference()
        )

        final_results["Context Precision"] = evaluate_metric(
            dataset=dataset,
            metric=context_precision_metric,
            metric_name="Context Precision",
            column_name="non_llm_context_precision_with_reference",
            evaluator_embeddings=evaluator_embeddings,
        )

        save_json(
            RESULTS_PATH,
            final_results,
        )

    except Exception as error:

        print("\nContext Precision failed:")
        print(repr(error))

        final_results["Context Precision"] = {
            "error": repr(error)
        }

        save_json(
            RESULTS_PATH,
            final_results,
        )

    # --------------------------------------------------
    # 3. Context Recall
    #
    # Non-LLM metric: no Qwen evaluation required.
    # --------------------------------------------------

    try:

        context_recall_metric = NonLLMContextRecall()

        final_results["Context Recall"] = evaluate_metric(
            dataset=dataset,
            metric=context_recall_metric,
            metric_name="Context Recall",
            column_name="non_llm_context_recall",
            evaluator_embeddings=evaluator_embeddings,
        )

        save_json(
            RESULTS_PATH,
            final_results,
        )

    except Exception as error:

        print("\nContext Recall failed:")
        print(repr(error))

        final_results["Context Recall"] = {
            "error": repr(error)
        }

        save_json(
            RESULTS_PATH,
            final_results,
        )

    # --------------------------------------------------
    # 4. Answer Relevancy
    #
    # This metric still requires an LLM judge.
    # We run it separately so the other three metrics
    # are not blocked by the local Qwen evaluator.
    # --------------------------------------------------

    try:

        print("\nLoading evaluator LLM for Answer Relevancy...")

        evaluator_llm = LangchainLLMWrapper(
            get_llm()
        )

        print("Evaluator LLM loaded successfully.")

        answer_relevancy_metric = ResponseRelevancy()

        final_results["Answer Relevancy"] = evaluate_metric(
            dataset=dataset,
            metric=answer_relevancy_metric,
            metric_name="Answer Relevancy",
            column_name="answer_relevancy",
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
        )

        save_json(
            RESULTS_PATH,
            final_results,
        )

    except Exception as error:

        print("\nAnswer Relevancy failed:")
        print(repr(error))

        final_results["Answer Relevancy"] = {
            "error": repr(error)
        }

        save_json(
            RESULTS_PATH,
            final_results,
        )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n==============================================")
    print("TASK 16 RAGAS EVALUATION FINISHED")
    print("==============================================")

    for metric_name, result in final_results.items():

        if not isinstance(result, dict):
            continue

        average = result.get("average")

        if average is not None:

            print(
                f"{metric_name}: "
                f"{average:.4f} "
                f"({result.get('evaluated_samples')}/"
                f"{result.get('total_samples')})"
            )

    print("\nFiles:")
    print(f"Input:   {INPUT_PATH}")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()






