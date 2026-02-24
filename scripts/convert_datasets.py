import json
import os
from pathlib import Path

import requests
from datasets import load_dataset

BASE = Path(__file__).parent.resolve() / ".." / "benchmarks" / "datasets"
os.chdir(BASE)


def obtain_sharegpt_dataset():

    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    response = requests.get(url)
    response.raise_for_status()
    with open("ShareGPT.json", "wb") as f:
        f.write(response.content)

def obtain_arena_dataset():

    ds = load_dataset("lmsys/chatbot_arena_conversations")

    chats = []

    for idx, item in enumerate(ds['train']):
        conv = item['conversation_a']
        question = conv[0]['content']
        answer = conv[1]['content']

        chats.append({
            'id': idx * 2,
            'conversations': [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        })

        conv = item['conversation_b']
        question = conv[0]['content']
        answer = conv[1]['content']

        chats.append({
            'id': idx * 2 + 1,
            'conversations': [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        })

    with open("arena.json", "w") as f:
        json.dump(chats, f, indent=2)


def obtain_spec_bench_dataset():
    datas = []

    url = "https://github.com/hemingkx/Spec-Bench/raw/refs/heads/main/data/spec_bench/question.jsonl"
    if not os.path.exists("questions.jsonl"):
        response = requests.get(url)
        response.raise_for_status()
        with open("questions.jsonl", "wb") as f:
            f.write(response.content)

    with open("questions.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            datas.append(data)
    os.remove("questions.jsonl")

    chats = []

    for data in datas:
        chats.append({
            'id': data['question_id'],
            'conversations': [
                {
                    "from": "human",
                    "value": data['turns'][0]
                },
                {
                    "from": "gpt",
                    "value": ""
                }
            ]
        })


    with open("spec_bench.json", "w") as f:
        json.dump(chats, f, indent=2)

def obtain_domain_tough_dataset():

    ds = load_dataset("YAV-AI/llm-domain-specific-tough-questions")

    chats = []

    for idx, item in enumerate(ds["train"]):
        chats.append({
            'id': idx,
            'conversations': [
                {
                    "from": "human",
                    "value": item["question"]
                },
                {
                    "from": "gpt",
                    "value": ""
                }
            ]
        })

    with open("tough.json", "w") as f:
        json.dump(chats, f, indent=2)


if __name__ == "__main__":
    obtain_sharegpt_dataset()
    obtain_arena_dataset()
    obtain_spec_bench_dataset()
    obtain_domain_tough_dataset()