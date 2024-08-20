"""Script to extract results from Langfuse API and save them to a CSV file."""

import logging
import os

import pandas as pd
import requests
from langfuse import Langfuse
from requests.auth import HTTPBasicAuth

from src.constants import evaluation_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URL = os.environ.get("LANGFUSE_SCORE_URL")
USERNAME = os.environ.get("LANGFUSE_PUBLIC_KEY")
PASSWORD = os.environ.get("LANGFUSE_SECRET_KEY")
PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
HOST = os.environ.get("LANGFUSE_HOST")
COLUMNS = evaluation_config.RESULT_COLUMNS


def get_score(score_id: str) -> tuple[str, float]:
    """
    Get the name and value of a score from the Langfuse API.

    Args:
        score_id (str): The ID of the score.

    Returns:
        Tuple[str, float]: The name and value of the score
    """
    response = requests.get(
        URL + score_id, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=60
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


def fetch_traces(client: Langfuse, local_session_ids: list[str]) -> list:
    """
    Fetch the traces from the Langfuse API.

    Args:
        client (Langfuse): The Langfuse client.
        local_session_ids (List[str]): The session IDs.

    Returns:
        List: The traces.
    """
    local_traces = []
    for session_id in local_session_ids:
        local_traces += client.fetch_traces(session_id=session_id).data
    return local_traces


def process_traces(local_traces: list) -> pd.DataFrame:
    """
    Process the traces and return a DataFrame.

    Args:
        local_traces (List): The traces.

    Returns:
        pd.DataFrame: The DataFrame.
    """
    data = []
    for trace in local_traces:
        experiment_name = trace.session_id
        trace_id = trace.id
        if "mixture" in experiment_name:
            question = trace.input["question"]
            answer = trace.output["content"]
        else:
            question = trace.input["query"]
            answer = trace.output["result"]

        scores = {
            score_name: score_value
            for score_id in trace.scores
            for score_name, score_value in [get_score(score_id)]
        }

        data.append(
            [
                experiment_name,
                trace_id,
                question,
                answer,
                scores["faithfulness"],
                scores["answer_relevancy"],
                scores["context_utilization"],
            ]
        )
    return pd.DataFrame(data, columns=COLUMNS)


if __name__ == "__main__":
    logger.info("Creating Langfuse client")
    langfuse_client = Langfuse(public_key=PUBLIC_KEY, secret_key=SECRET_KEY, host=HOST)
    logger.info("Fetching session IDs - experiment names")
    session_ids = fetch_session_ids(langfuse_client)
    logger.info("Session IDs fetched")
    logger.info("Fetching traces")
    traces = fetch_traces(langfuse_client, session_ids)
    logger.info("Traces fetched")
    logger.info("Processing traces - extracting scores from traces")
    dataframe = process_traces(traces)
    logger.info("Traces processed")
    os.makedirs("results", exist_ok=True)
    logger.info("Saving results to CSV")
    dataframe.to_csv("./results/results.csv", index=False)
    logger.info("Results saved to CSV")
