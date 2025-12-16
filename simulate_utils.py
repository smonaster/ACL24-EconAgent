import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import multiprocessing
import scipy
import time as pytime
import requests
from functools import partial

save_path = './'

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://localhost:8000/predict")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "200"))

def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def _format_dialog_history(dialogs, mode="plain"):
    """
    Format dialog history for the LLM payload.
    mode="plain": ROLE: content per line (legacy/default).
    mode="chat_tags": tag roles to mimic chat-style prompts in a single string.
    """
    if mode == "chat_tags":
        chunks = []
        for message in dialogs:
            role = message.get("role", "user").lower()
            content = message.get("content", "")
            if role == "system":
                chunks.append(f"<|system|>\n{content}")
            elif role == "assistant":
                chunks.append(f"<|assistant|>\n{content}")
            else:
                chunks.append(f"<|user|>\n{content}")
        return "\n".join(chunks) + "\n<|assistant|>\n"

    # Default legacy format
    turns = []
    for message in dialogs:
        role = message.get('role', 'user').upper()
        content = message.get('content', '')
        turns.append(f"{role}: {content}")
    return "\n".join(turns)


def _extract_response(data):
    for key in ('respuesta', 'response', 'content', 'message'):
        if key in data and isinstance(data[key], str):
            text = data[key].replace('[INST]', '').replace('[/INST]', '').strip()
            return text
    raise ValueError("LLM endpoint response missing text content")


def _call_llm(payload):
    response = requests.post(
        LLM_ENDPOINT,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=LLM_TIMEOUT,
    )
    response.raise_for_status()
    return _extract_response(response.json())


def get_multiple_completion(dialogs, num_cpus=1, temperature=0, max_tokens=None, format_mode="plain"):
    get_completion_partial = partial(
        get_completion,
        temperature=temperature,
        max_tokens=max_tokens,
        format_mode=format_mode,
    )
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(get_completion_partial, dialogs)
    return results


def get_completion(dialogs, temperature=0, max_tokens=None, format_mode="plain"):
    payload = {
        "message": _format_dialog_history(dialogs, mode=format_mode),
        "max_tokens": max_tokens if max_tokens is not None else LLM_MAX_TOKENS,
        "temperature": temperature,
    }
    max_retries = 20
    for i in range(max_retries):
        try:
            return _call_llm(payload)
        except Exception as e:
            if i < max_retries - 1:
                pytime.sleep(6)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "Error"

def format_numbers(numbers):
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers):
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'
