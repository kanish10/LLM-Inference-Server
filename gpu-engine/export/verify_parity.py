"""Generate HF fp32 greedy-decode reference outputs for the M1 CPU-oracle parity test.

Usage:
    venv/bin/python verify_parity.py --out ../models/parity_reference.json

Ten fixed prompts, greedy (temperature=0) decode of 32 new tokens each, run in
fp32 on CPU through the real HuggingFace gpt2 model. The C++ CPU oracle
(reference/cpu_model.cpp) must reproduce the same token IDs for M1 to pass.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROMPTS = [
    "The quick brown fox",
    "In a shocking turn of events,",
    "Once upon a time, there was a",
    "The capital of France is",
    "def fibonacci(n):",
    "Scientists have discovered that",
    "To be or not to be, that is",
    "The stock market today",
    "My favorite recipe for pasta involves",
    "The history of artificial intelligence began",
    # Widened set (M2 GPU-vs-CPU parity needs more samples than M1's original
    # 10 for a statistically meaningful match-rate read -- see VERIFICATION.md).
    "The president announced that",
    "In the beginning, God created",
    "class BinarySearchTree:",
    "The weather forecast for tomorrow",
    "According to the latest research,",
    "She opened the door and saw",
    "The recipe calls for two cups of",
    "import numpy as np",
    "The meaning of life is",
    "On a dark and stormy night,",
    "The United Nations announced today",
    "def quicksort(arr):",
    "The best way to learn programming is",
    "He looked up at the stars and",
    "The economic outlook for next year",
    "SELECT * FROM users WHERE",
    "The ancient city of Rome was",
    "Water boils at a temperature of",
    "The novel begins with a description of",
    "In physics, the theory of relativity",
    "The chef added a pinch of salt and",
    "Machine learning models are trained by",
    "The mountain climbers reached the summit",
    "The Great Wall of China was built",
    "def merge_sort(arr):",
    "The detective examined the crime scene",
    "The company reported quarterly earnings of",
    "The sun rises in the",
    "A long time ago in a galaxy",
    "The professor explained that photosynthesis",
    "The election results showed that",
]

NUM_NEW_TOKENS = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="../models/parity_reference.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval().to(torch.float32)

    cases = []
    for prompt in PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=NUM_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        prompt_tokens = input_ids[0].tolist()
        generated_tokens = out[0, input_ids.shape[1]:].tolist()
        cases.append(
            {
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            }
        )
        print(f"{prompt!r} -> {tokenizer.decode(generated_tokens)!r}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"num_new_tokens": NUM_NEW_TOKENS, "cases": cases}, f, indent=2)
    print(f"\nWrote {out_path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
