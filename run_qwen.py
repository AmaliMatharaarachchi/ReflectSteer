"""Run Qwen2.5-7B-Instruct locally using Hugging Face Transformers.

Use a locally downloaded checkpoint (see README.md for the NCI environment).

Examples:
    python run_qwen.py "Explain reinforcement learning in three sentences."
    python run_qwen.py "Write a short poem." --max-new-tokens 128
    python run_qwen.py "Hello!" --device cpu

Inference uses only downloaded local files and requires no internet access.
The full model requires substantial memory; CPU inference can be slow.
"""

import argparse
from pathlib import Path


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Give me a short introduction to large language models.")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--temperature", type=float, default=0.7, help="Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("prompt must not be empty")
    if args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def main():
    from qwen_llm import QwenLM
    args = parse_args()
    lm = QwenLM(args.model_dir, args.device, args.max_input_tokens, args.max_new_tokens)
    result = lm.generate([
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ], temperature=args.temperature, top_p=args.top_p)
    print(result['text'])


if __name__ == "__main__":
    main()
