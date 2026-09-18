import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
)
from ragas.run_config import RunConfig

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "evaluation" / "ragas_input.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    "user_input": data["question"],
    "response": data["answer"],
    "retrieved_contexts": data["contexts"],
    "reference_contexts": [
        [context]
        for context in data["ground_truth"]
    ],
})

print("Loaded:", len(dataset), "questions")

metrics = [
    NonLLMContextPrecisionWithReference(),
    NonLLMContextRecall(),
]

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    batch_size=1,
    run_config=RunConfig(
        timeout=30,
        max_retries=0,
        max_workers=1,
    ),
    raise_exceptions=False,
    show_progress=True,
)

df = result.to_pandas()

print("\nRESULT COLUMNS:")
print(list(df.columns))

print("\nRESULTS:")
print(df.to_string())

for column in [
    "non_llm_context_precision_with_reference",
    "non_llm_context_recall",
]:
    if column in df.columns:
        values = [
            float(x)
            for x in df[column]
            if x is not None and str(x).lower() != "nan"
        ]

        print(f"\n{column}")
        print("Valid:", len(values), "/", len(df))

        if values:
            print("Average:", sum(values) / len(values))
