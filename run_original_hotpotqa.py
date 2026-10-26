"""Run original Reflexion HotPotQA notebooks' agents with local Qwen."""
import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=['cot-context', 'cot-no-context', 'react'], default='cot-context')
    parser.add_argument('--strategy', choices=['base', 'reflexion', 'last_trial', 'last_trial_and_reflexion'], default='reflexion')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/Qwen2.5-7B-Instruct')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/hotpotqa_sample.json')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=6, help='Original ReAct step budget; unused by CoT')
    parser.add_argument('--max-input-tokens', type=int, default=16384, help='Fail rather than truncate upstream prompts')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.limit, args.trials, args.max_steps, args.max_input_tokens) < 1 or args.offset < 0:
        parser.error('Budgets must be positive; offset must be nonnegative')
    os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT / 'data/tiktoken'))
    # Resolve all agent/framework imports before loading the NCI PyTorch stack.
    from original_hotpotqa import make_agents, run_trials, source_hashes, upstream
    upstream()
    from qwen_llm import QwenLM
    rows = json.loads(args.data.read_text(encoding='utf-8'))[args.offset:args.offset + args.limit]
    if not rows:
        parser.error('Selected dataset range is empty')
    if args.agent == 'cot-context' and any('supporting_facts' not in row for row in rows):
        parser.error('Regenerate data with prepare_hotpotqa.py; supporting_facts are required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        def write(event):
            output.write(json.dumps(event, ensure_ascii=False) + '\n')
            output.flush()
            if event['type'] == 'trial_summary':
                print(json.dumps(event), flush=True)

        lm = QwenLM(args.model_dir, args.device, args.max_input_tokens, 250)
        agents = make_agents(rows, lm, args.agent, args.strategy, args.max_steps)
        write({'type': 'metadata', 'implementation': 'upstream_agents',
               'upstream_commit': '218cf0ef1df84b05ce379dd4a8e47f17766733a0',
               'source_sha256': source_hashes(), 'model': lm.metadata,
               'packages': {name: version(name) for name in
                            ('torch', 'transformers', 'langchain', 'langgraph',
                             'langchain-classic', 'langchain-community', 'tiktoken', 'wikipedia')},
               'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
               'question_ids': [row['id'] for row in rows],
               'retrieval': 'live_wikipedia' if args.agent == 'react' else args.agent,
               'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
        run_trials(agents, args.agent, args.strategy, args.trials, write)


if __name__ == '__main__':
    main()
