"""Convert the upstream trusted HotPotQA joblib sample to portable JSON."""
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent


def main():
    import joblib
    import pandas as pd
    # The upstream sample references numeric index classes removed in pandas 2.
    name = 'pandas.core.indexes.numeric'
    previous = sys.modules.get(name)
    shim = types.ModuleType(name)
    shim.Int64Index = shim.UInt64Index = shim.Float64Index = pd.Index
    sys.modules[name] = shim
    try:
        data = joblib.load(ROOT / 'Reflexion/hotpotqa_runs/data/hotpot-qa-distractor-sample.joblib')
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    rows = []
    for row in data.to_dict(orient='records'):
        context = row['context']
        rows.append({'id': str(row['id']), 'question': row['question'], 'answer': row['answer'],
                     'supporting_facts': {'title': list(row['supporting_facts']['title']),
                                          'sent_id': [int(i) for i in row['supporting_facts']['sent_id']]},
                     'context': [[title, list(sentences)] for title, sentences in
                                 zip(context['title'], context['sentences'])]})
    target = ROOT / 'data/hotpotqa_sample.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Converted {len(rows)} questions to {target}')


if __name__ == '__main__':
    main()
