"""Constants connected to evaluation of the model."""

from ragas.metrics import answer_relevancy, context_utilization, faithfulness

LLM_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

LLM_MODEL_KWARGS = {"max_tokens": 4096, "temperature": 0, "top_p": 0}

EMBEDDER_MODEL_ID = "amazon.titan-embed-text-v2:0"

EMBEDDER_MODEL_KWARGS = {"dimensions": 512, "normalize": True}

METRICS = [faithfulness, answer_relevancy, context_utilization]

RESULT_COLUMNS = [
    "experiment_name",
    "trace_id",
    "question",
    "answer",
    "faithfulness",
    "answer_relevancy",
    "context_utilization",
]
