# Original Reflexion HotPotQA with local Qwen

`run_original_hotpotqa.py` calls the original `CoTAgent`, `ReactAgent` and
`ReactReflectAgent` in `reflexion_hotpotqa/agents.py`. Qwen2.5-7B-Instruct
replaces the OpenAI backend through a LangChain LLM adapter. LangGraph schedules
trial rounds; the original classes perform reasoning, actions and reflection.

The runner follows the upstream notebooks: the same 100-question sample, original
few-shot prompts, exact-match normalization, five trials, and retries only for
unsolved questions. It reports cumulative exact-match accuracy after every trial.
This replicates the method with Qwen; it does not reproduce the GPT model scores.

## NCI Gadi

Copy the whole checkout, including the bundled `reflexion_hotpotqa/` package and `data/`,
to Gadi. The PBS script reuses the downloaded Qwen checkpoint and uses a separate
ReflectSteer runtime environment. With internet access, create it and install
the dependencies once (the system site packages expose NCI's PyTorch):

```bash
module load pytorch/2.12.0
export PYTHONNOUSERSITE=1
python3 -m venv --system-site-packages /scratch/qw14/aa1215/ReflectSteer/venv-nci
source /scratch/qw14/aa1215/ReflectSteer/venv-nci/bin/activate
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
cd /home/572/aa1215/workspace/repos/ReflectSteer
python3 -m pip install -r requirements.txt
```

Start with one question and one trial:

```bash
qsub -v HOTPOT_LIMIT=1,HOTPOT_TRIALS=1 run_hotpotqa.pbs
```

Then run the original **CoT with context** baseline and Reflexion conditions:

```bash
qsub -v HOTPOT_STRATEGY=base run_hotpotqa.pbs
qsub -v HOTPOT_STRATEGY=reflexion run_hotpotqa.pbs
```

Both run all 100 sampled questions for five trials by default. The job requests
one V100, 12 CPUs, 64 GB memory and two hours; measure the smoke run and adjust
walltime for the full experiment. Project `qw14` and storage/environment paths
come from AgentRL. Override paths using `QWEN_VENV` and `QWEN_MODEL_DIR` via
`qsub -v` if needed. Compute inference uses local checkpoint files.

The script writes to `/home/572/aa1215/workspace/repos/ReflectSteer/outs/`
regardless of the submission directory. Override `REFLECTSTEER_REPO` via `qsub -v`
if your checkout is elsewhere. A per-job `.log` is created there before loading
modules or activating the environment; setup failures include the failing line.
PBS also retains its normal `hotpotqa_reflexion.o<job-id>` log. A finished PBS job
can have failed: check these logs and the exit status before interpreting results.

The original CoT-context notebook supplies **gold supporting paragraphs**,
including repeated article titles in supporting-fact order. This is the original
oracle-context setting, not retrieval over all distractors. CoT without context
uses the original separate prompt and examples:

```bash
qsub -v HOTPOT_AGENT=cot-no-context,HOTPOT_STRATEGY=base run_hotpotqa.pbs
qsub -v HOTPOT_AGENT=cot-no-context,HOTPOT_STRATEGY=reflexion run_hotpotqa.pbs
```

For direct execution in an allocated GPU session:

```bash
python run_original_hotpotqa.py --agent cot-context --strategy reflexion \
  --model-dir /scratch/qw14/aa1215/qwen/model \
  --limit 1 --trials 1 --output outs/original_smoke.jsonl
```

## Original ReAct requires Wikipedia access

`--agent react` uses the original Wikipedia docstore and Search/Lookup behavior,
six steps per attempt, and the original 3896-token halt rule measured with
`tiktoken` p50k_base. It never substitutes distractor passages. Run it where the
GPU process can access Wikipedia, or configure network access before using it on
NCI. The offline PBS default is CoT-context. A live Wikipedia run also reflects
changes to Wikipedia since the paper.

```bash
python run_original_hotpotqa.py --agent react --strategy reflexion \
  --model-dir /path/to/local/qwen --limit 100 --trials 5 \
  --output outs/react_reflexion.jsonl
```

## Preserved behavior and compatibility changes

- Prompts and few-shot strings are unchanged. Thought and action remain separate
  LLM calls. Each upstream prompt becomes one user chat message, matching the
  original `AnyOpenAILLM` chat path, using Qwen's chat template.
- Decoding is greedy. ReAct thought/action calls allow 100 new tokens and stop
  at newline; ReAct reflections allow 250 tokens without a newline stop. CoT
  thought, action and reflection calls allow 250 tokens with a newline stop.
- All four upstream strategies are available: `base`, `reflexion`, `last_trial`,
  `last_trial_and_reflexion`. Correct questions remain frozen on later trials.
- Imports use modern LangChain core/classic/community namespaces. OpenAI models
  are no longer instantiated at import time; runners inject Qwen explicitly.
  Package-relative imports support direct use of the cloned source.
- One upstream bug is corrected: malformed actions return `(None, None)` instead
  of crashing during tuple unpacking. Original invalid-action feedback is retained.
- Original prompt and scratchpad handling remain in the upstream agents. The
  Qwen adapter never silently drops history; it fails if a prompt exceeds its
  16384-token input safety limit. Qwen tokens and the original ReAct halt counter
  use different tokenizers, as recorded here.

`outs/hotpotqa.<job-id>.jsonl` in the ReflectSteer repository contains metadata, each attempted question's
scratchpad, reflections, exact model prompts/responses and token counts, and
per-trial cumulative accuracy. Metadata records the source hashes, dataset hash,
question IDs, checkpoint path and run configuration. Prior files are protected
against overwrite. Trial summaries include already-solved questions in the
original fixed denominator.

The bundled JSON preserves the sample's questions, answers, context and supporting
facts. `data/tiktoken/` contains the p50k_base vocabulary so the original counter
loads without downloads. Regenerate JSON with `python prepare_hotpotqa.py` after
installing `pandas` and `joblib`; copy the regenerated data to NCI as well.

## Validation and earlier experimental runner

```bash
python -m unittest discover -s tests -v
```

Tests exercise the actual upstream classes with scripted model responses, including
prompt identity, context selection, reflection, trial order, success freezing,
original ReAct docstore execution and invalid-action handling. GPU inference and
full benchmark numbers still need to be measured on NCI.

The earlier custom offline-passage LangGraph agent remains in `hotpotqa_agent.py`
and `run_hotpotqa_qwen.py` for comparison. `run_hotpotqa.pbs` now uses the original
implementation instead.
