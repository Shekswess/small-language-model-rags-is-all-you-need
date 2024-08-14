"""Script to extract results from Langfuse API and save them to a CSV file."""

import os

import pandas as pd
import requests
from langfuse import Langfuse
from requests.auth import HTTPBasicAuth

URL = "http://cloud.langfuse.com/api/public/scores/"
USERNAME = os.environ.get("LANGFUSE_PUBLIC_KEY")
PASSWORD = os.environ.get("LANGFUSE_SECRET_KEY")
PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
HOST = os.environ.get("LANGFUSE_HOST")
COLUMNS = [
    "experiment_name",
    "trace_id",
    "faithfulness",
    "answer_relevancy",
    "context_utilization",
]


def get_score(score_id: str) -> tuple[str, float]:
    """
    Get the name and value of a score from the Langfuse API.

    Args:
        score_id (str): The ID of the score.

    Returns:
        Tuple[str, float]: The name and value of the score
    """
    response = requests.get(
        URL + score_id, auth=HTTPBasicAuth(USERNAME, PASSWORD)
    ).json()
    score_name = response["name"]
    score_value = response["value"]
    return score_name, score_value


def fetch_session_ids(client: Langfuse) -> list[str]:
    """
    Fetch the session IDs from the Langfuse API.

    Args:
        client (Langfuse): The Langfuse client.

    Returns:
        List[str]: The session IDs.
    """
    return [
        session_id.id
        for session_id in client.fetch_sessions().data
        if "mixture-rag" in session_id.id
        or "simple-rag" in session_id.id
        and "simple-rag-mixtral-8x7b" != session_id.id
    ]


def fetch_traces(client: Langfuse, session_ids: list[str]) -> list:
    """
    Fetch the traces from the Langfuse API.

    Args:
        client (Langfuse): The Langfuse client.
        session_ids (List[str]): The session IDs.

    Returns:
        List: The traces.
    """
    traces = []
    for session_id in session_ids:
        traces += client.fetch_traces(session_id=session_id).data
    return traces


def process_traces(traces: list) -> pd.DataFrame:
    """
    Process the traces and return a DataFrame.

    Args:
        traces (List): The traces.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    data = []
    for trace in traces:
        experiment_name = trace.session_id
        trace_id = trace.id
        trace_scores_ids = trace.scores
        scores = {}
        for score_id in trace_scores_ids:
            score_name, score_value = get_score(score_id)
            scores[score_name] = score_value
        faithfulness = scores["faithfulness"]
        answer_relevancy = scores["answer_relevancy"]
        context_utilization = scores["context_utilization"]
        data.append(
            [
                experiment_name,
                trace_id,
                faithfulness,
                answer_relevancy,
                context_utilization,
            ]
        )
    return pd.DataFrame(data, columns=COLUMNS)


if __name__ == "__main__":
    langfuse_client = Langfuse(public_key=PUBLIC_KEY, secret_key=SECRET_KEY, host=HOST)
    session_ids = fetch_session_ids(langfuse_client)
    traces = fetch_traces(langfuse_client, session_ids)
    dataframe = process_traces(traces)
    os.makedirs("results", exist_ok=True)
    dataframe.to_csv("./results/results.csv", index=False)
