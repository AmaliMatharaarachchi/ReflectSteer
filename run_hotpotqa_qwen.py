"""Run local Qwen on the upstream Reflexion HotPotQA sample without network access."""
import argparse
import json
from pathlib import Path
from hotpotqa_agent import run_question
from qwen_llm import QwenLM

ROOT = Path(__file__).resolve().parent


def load_rows(path):
    if path.suffix == '.json':
        rows = json.loads(path.read_text(encoding='utf-8'))
    elif path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        import joblib
        data = joblib.load(path)
        rows = data.to_dict(orient='records') if hasattr(data, 'to_dict') else data
    if not isinstance(rows, list) or not rows:
        raise ValueError('Dataset must contain a nonempty list of questions')
    for row in rows:
        for key in ('question', 'answer', 'context'):
            if key not in row:
                raise ValueError(f'Dataset row is missing {key}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/Qwen2.5-7B-Instruct')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/hotpotqa_sample.json')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--no-memory', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.limit, args.attempts, args.max_steps, args.max_input_tokens, args.max_new_tokens) < 1 or args.offset < 0:
        parser.error('Budgets must be positive and offset nonnegative')
    rows = load_rows(args.data)[args.offset:args.offset + args.limit]
    if not rows:
        parser.error('Selected dataset range is empty')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects prior experiments from accidental overwrite.
    with args.output.open('x', encoding='utf-8') as output:
        lm = QwenLM(args.model_dir, args.device, args.max_input_tokens, args.max_new_tokens)
        metadata = {'type': 'metadata', 'retrieval': 'offline_distractor_passages',
                    'agent': 'langgraph_reflexion', 'action_protocol': 'bracket_actions_to_langchain_tools',
                    'model': lm.metadata, 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        output.write(json.dumps(metadata) + '\n')
        output.flush()
        for index, row in enumerate(rows, start=args.offset):
            result = run_question(lm, row, args.attempts, args.max_steps, not args.no_memory, args.trace)
            result.update(type='question', index=index)
            output.write(json.dumps(result) + '\n')
            output.flush()
            print(f'Question {index}: success={result["success"]}, attempts={len(result["trials"])}', flush=True)


if __name__ == '__main__':
    main()
