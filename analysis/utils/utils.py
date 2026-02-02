import json
import os   
import random
import re
from typing import Dict

def load_json(path):
    print(f"Loading JSON from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    print(f"Saving JSON to {path}...")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def flip_answer(answer):
    answer = answer.strip()
    # Boolean flip
    if answer.lower() == 'yes':
        return 'No'
    elif answer.lower() == 'no':
        return 'Yes'
    # Single-choice A/B/C flip
    if answer.upper() in {'A', 'B', 'C'}:
        pool = {'A', 'B', 'C'} - {answer.upper()}
        return random.choice(list(pool))
    return answer  # fallback for anything else

def parse_model_output(text: str, enable_cot: bool) -> Dict[str, str]:
    if not enable_cot:
        return {"answer": text, "thought": None}
    
    pattern = r"<think>(.*?)</think>\s*(.*)"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return {
            "thought": match.group(1).strip(),
            "answer": match.group(2).strip()
        }
    else:
        if "<think>" in text:
            parts = text.split("</think>")
            if len(parts) > 1:
                 return {
                    "thought": parts[0].replace("<think>", "").strip(),
                    "answer": parts[1].strip()
                }
        return {"answer": text, "thought": "Failed to parse <think> tags"}

def parse_qwen_thinking(text: str) -> Dict[str, str]:
    # split after </think> token
    parts = text.split("</think>")
    if len(parts) > 1:
        return {
            "thought": parts[0].strip(),
            "answer": parts[1].strip()
        }
    return {"answer": text, "thought": "Failed to parse </think> tags"}