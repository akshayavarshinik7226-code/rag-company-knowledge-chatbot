"""RAGAS evaluation for the company knowledge chatbot."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

from backend.llm_factory import get_llm
from backend.rag_chain import ask_question


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "evaluation" / "test_dataset.json"
INPUT_PATH = BASE_DIR / "evaluation" / "ragas_input.json"
RESULTS_PATH = BASE_DIR / "evaluation" / "ragas_results.json"

MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def load_test_dataset() -> list[dict]:
    """Load the 10 evaluation questions and ground-truth answers."""
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict) -> None:
    """Save JSON data."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    """Load JSON data."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_ragas_dataset(data: dict) -> Dataset:
    """Convert saved RAG results into a HuggingFace Dataset."""
    return Dataset.from_dict(
        {
            "question": data["question"],
            "answer": data["answer"],
            "contexts": data["contexts"],
            "ground_truth": data["ground_truth"],
        }
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    # -----------------------------------------------------
    # Load the 10-question evaluation dataset
    # -----------------------------------------------------

    test_data = load_test_dataset()

    print(f"Loaded {len(test_data)} evaluation questions.")


    # -----------------------------------------------------
    # STEP 1:
    # Run the RAG pipeline ONLY if results were not already
    # saved from an earlier run.
    # -----------------------------------------------------

    if INPUT_PATH.exists():

        print("\n==============================================")
        print("Loading previously saved RAG evaluation data")
        print("==============================================")

        saved_data = load_json(INPUT_PATH)

        print(
            f"Loaded saved results for "
            f"{len(saved_data['question'])} questions."
        )

    else:

        print("\n==============================================")
        print("Running RAG pipeline for 10 questions")
        print("==============================================")

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for index, item in enumerate(test_data, start=1):

            question = item["question"]

            print(
                f"\n[{index}/{len(test_data)}] "
                f"{question}"
            )

            try:
                result = ask_question(question)

                answer = result.get("answer", "")

                retrieved_contexts = [
                    context.get("text", "")
                    for context in result.get("sources", [])
                    if context.get("text")
                ]

                questions.append(question)
                answers.append(answer)
                contexts.append(retrieved_contexts)
                ground_truths.append(item["ground_truth"])

                print(
                    f"    Answer: {answer[:180]}"
                )

                print(
                    f"    Contexts retrieved: "
                    f"{len(retrieved_contexts)}"
                )

            except Exception as error:

                print(
                    f"    RAG pipeline error: {error}"
                )

                questions.append(question)
                answers.append(
                    "I couldn't find that information "
                    "in the company knowledge base."
                )
                contexts.append([])
                ground_truths.append(item["ground_truth"])


        # Save the RAG pipeline results.
        saved_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        save_json(INPUT_PATH, saved_data)

        print("\n==============================================")
        print("RAG pipeline completed")
        print("==============================================")

        print(
            f"Saved RAG evaluation data to:\n"
            f"{INPUT_PATH}"
        )


    # -----------------------------------------------------
    # Create RAGAS dataset
    # -----------------------------------------------------

    dataset = create_ragas_dataset(saved_data)


    # -----------------------------------------------------
    # Load local HuggingFace embeddings
    # -----------------------------------------------------

    print("\nLoading local HuggingFace embeddings...")

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=str(MODEL_PATH)
        )
    )

    print("Evaluator embeddings loaded successfully.")


    # -----------------------------------------------------
    # RAGAS metrics
    # -----------------------------------------------------

    metrics = [
        (
            "Faithfulness",
            Faithfulness(),
            "faithfulness",
        ),
        (
            "Answer Relevancy",
            ResponseRelevancy(),
            "answer_relevancy",
        ),
        (
            "Context Precision",
            LLMContextPrecisionWithReference(),
            "context_precision",
        ),
        (
            "Context Recall",
            LLMContextRecall(),
            "context_recall",
        ),
    ]


    # -----------------------------------------------------
    # Load previous metric results if available
    # -----------------------------------------------------

    if RESULTS_PATH.exists():

        try:
            final_results = load_json(RESULTS_PATH)

        except Exception:
            final_results = {}

    else:
        final_results = {}


    # -----------------------------------------------------
    # RAGAS configuration
    # -----------------------------------------------------

    run_config = RunConfig(
        timeout=180,
        max_retries=0,
        max_workers=1,
    )


    # -----------------------------------------------------
    # Evaluate each metric separately
    # -----------------------------------------------------

    for metric_name, metric, column_name in metrics:

        # Do not repeat a metric that was already saved.
        if metric_name in final_results:

            print(
                f"\n{metric_name} already has saved results."
            )

            print("Skipping this metric.")

            continue


        print("\n==============================================")
        print(f"Evaluating: {metric_name}")
        print("==============================================")


        try:

            # Create a fresh evaluator LLM for every metric.
            # This helps avoid asyncio/event-loop problems
            # when RAGAS runs multiple evaluations.
            print("Loading evaluator LLM...")

            evaluator_llm = LangchainLLMWrapper(
                get_llm()
            )

            print("Evaluator LLM loaded successfully.")


            # Run one metric.
            result = evaluate(
                dataset=dataset,
                metrics=[metric],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                batch_size=1,
                run_config=run_config,
                raise_exceptions=False,
                show_progress=True,
            )


            # Convert result to DataFrame.
            dataframe = result.to_pandas()


            print(
                "\nRAGAS result columns:"
            )

            print(
                list(dataframe.columns)
            )


            # -------------------------------------------------
            # Find the metric column.
            # -------------------------------------------------

            if column_name not in dataframe.columns:

                print(
                    f"\nCould not find expected column: "
                    f"{column_name}"
                )

                print(
                    "Available columns:",
                    list(dataframe.columns),
                )

                continue


            # Get individual scores.
            raw_values = dataframe[column_name].tolist()


            # Convert valid numerical values.
            scores = []

            for value in raw_values:

                try:

                    if value is None:
                        continue

                    score = float(value)

                    # Ignore NaN and infinity.
                    if score != score:
                        continue

                    scores.append(score)

                except (TypeError, ValueError):

                    continue


            # Calculate average.
            if scores:

                average = sum(scores) / len(scores)

            else:

                average = None


            # Save the metric.
            final_results[metric_name] = {
                "scores": raw_values,
                "valid_scores": scores,
                "average": average,
                "evaluated_samples": len(scores),
                "total_samples": len(raw_values),
            }


            # IMPORTANT:
            # Save immediately after every metric.
            save_json(
                RESULTS_PATH,
                final_results,
            )


            print(
                f"\n{metric_name} completed."
            )

            if average is not None:

                print(
                    f"Average score: {average:.4f}"
                )

            else:

                print(
                    "Average score: No valid scores"
                )

            print(
                f"Valid samples: "
                f"{len(scores)}/{len(raw_values)}"
            )

            print(
                f"Saved to:\n{RESULTS_PATH}"
            )


        except Exception as error:

            print(
                f"\n{metric_name} failed:"
            )

            print(
                repr(error)
            )

            print(
                "Continuing with the next metric..."
            )


    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n==============================================")
    print("RAGAS evaluation finished")
    print("==============================================")


    if final_results:

        for metric_name, result in final_results.items():

            average = result.get("average")

            if average is not None:

                print(
                    f"{metric_name}: "
                    f"{average:.4f}"
                )

            else:

                print(
                    f"{metric_name}: "
                    f"No valid score"
                )


    print("\nFiles:")

    print(
        f"RAGAS input:\n{INPUT_PATH}"
    )

    print(
        f"RAGAS results:\n{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()