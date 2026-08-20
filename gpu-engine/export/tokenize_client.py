"""BPE encode/decode helper for demos. The engine itself speaks token IDs
only (see include/engine/engine_api.h); this wraps HF's GPT-2 tokenizer so
a human can feed text prompts in and get text back out on the client side.

Usage:
    venv/bin/python tokenize_client.py encode "Hello, world"
    venv/bin/python tokenize_client.py decode 15496 11 995
"""
import argparse

from transformers import GPT2Tokenizer


class Tokenizer:
    def __init__(self, model: str = "gpt2"):
        self._tok = GPT2Tokenizer.from_pretrained(model)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode")
    enc.add_argument("text")

    dec = sub.add_parser("decode")
    dec.add_argument("token_ids", nargs="+", type=int)

    args = ap.parse_args()
    tok = Tokenizer(args.model)

    if args.cmd == "encode":
        print(" ".join(str(t) for t in tok.encode(args.text)))
    else:
        print(tok.decode(args.token_ids))
