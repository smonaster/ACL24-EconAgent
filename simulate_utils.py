import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import scipy
import time as pytime
import requests
import json
import dirtyjson as dj
from functools import partial
from concurrent.futures import ThreadPoolExecutor

save_path = './'

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://localhost:8000/predict")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def _extract_response(data):
    
    if isinstance(data, dict):
        if 'response' in data:
            return data['response']
        if 'content' in data: 
            return data['content']
            
    raise ValueError(f"Formato de respuesta LLM desconocido o inválido: {data}")


def _call_llm(payload):
    print(f"[LLM REQUEST] Messages: {len(payload.get('messages', []))} length")
    print(f"[LLM REQUEST BODY] {payload}")

    try:
        response = requests.post(
            LLM_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        response_json = response.json()
        print(f"[LLM RAW RESPONSE] {response_json}")
    except requests.exceptions.RequestException as e:
        print(f"[LLM ERROR] Connection failed: {e}")
        raise e
    return _extract_response(response_json)

def extract_json_object(text: str, index: int = 0, parse: bool = False):
    """
    Return the nth JSON-like object substring in the text (default last).
    If parse=True, attempt to parse that fragment with dirtyjson and return the object.
    """
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start:i + 1])
                start = None
    if not objects:
        return None
    idx = index if index >= 0 else len(objects) + index
    chosen = objects[-1] if idx < 0 or idx >= len(objects) else objects[idx]
    cleaned = re.sub(r"\s+", " ", chosen).strip()
    if not parse:
        return cleaned
    try:
        return dj.loads(cleaned)
    except dj.error.Error:
        return None

def get_multiple_completion(dialogs_batch, num_cpus=1, temperature=0, max_tokens=None):
    get_completion_partial = partial(
        get_completion,
        temperature=temperature,
        max_tokens=max_tokens
    )
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        return list(executor.map(get_completion_partial, dialogs_batch))

def get_completion(dialog_history, temperature=0, max_tokens=None, format_mode=None):
    payload = {
        "messages": dialog_history,
        "max_tokens": max_tokens if max_tokens is not None else LLM_MAX_TOKENS,
        "temperature": temperature,
    }

    max_retries = 20
    for i in range(max_retries):
        try:
            return _call_llm(payload)
        except Exception as e:
            if i < max_retries - 1:
                pytime.sleep(2 * (i + 1))
            else:
                print(f"FAILED after {max_retries} retries. Error: {e}")
                return "{}"

def format_numbers(numbers):
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers):
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'
