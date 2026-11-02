# Original Reflexion HotPotQA with local Qwen

`run_original_hotpotqa.py` calls the original `CoTAgent`, `ReactAgent` and
`ReactReflectAgent` in `Reflexion/hotpotqa_runs/agents.py`. Qwen2.5-7B-Instruct
replaces the OpenAI backend through a LangChain LLM adapter. LangGraph schedules
trial rounds; the original classes perform reasoning, actions and reflection.

The runner follows the upstream notebooks: the same 100-question sample, original
few-shot prompts, exact-match normalization, five trials, and retries only for
unsolved questions. It reports cumulative exact-match accuracy after every trial.
This replicates the method with Qwen; it does not reproduce the GPT model scores.

## NCI Gadi

`Reflexion/` is an unmodified Git submodule declared in `.gitmodules`, pinned to
commit `218cf0ef1df84b05ce379dd4a8e47f17766733a0`. After cloning or updating
this repository, initialize it:

```bash
git submodule update --init --recursive
```

Copy the whole initialized checkout, including `Reflexion/` and `data/`,
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

Our experiment compares **ReAct without reflection** (`base`) against
**ReAct + Reflexion** (`reflexion`). Both use Wikipedia Search/Lookup without
gold supporting context. Our method will be added later.
The GPU process must have Wikipedia access. The earlier CoT-context smoke test
did not exercise retrieval or reflection; verify a small ReAct run first.

Run the two conditions:

```bash
qsub -v HOTPOT_AGENT=react,HOTPOT_STRATEGY=base run_hotpotqa.pbs
qsub -v HOTPOT_AGENT=react,HOTPOT_STRATEGY=reflexion run_hotpotqa.pbs
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
python run_original_hotpotqa.py --agent react --strategy reflexion \
  --model-dir /scratch/qw14/aa1215/qwen/model \
  --limit 1 --trials 1 --output outs/original_smoke.jsonl
```

## Original ReAct requires Wikipedia access

### Shared-storage Wikipedia relay on Gadi

The GPU job can delegate live Wikipedia searches to a separate internet-enabled
`copyq` worker. This is an experimental deployment: check NCI permits this use of
`copyq` before running it. Only retrieval runs there; model inference stays on the
GPU. Both jobs must see the same directory on `/scratch/qw14`.

After updating this checkout and initializing the submodule, submit one worker
and two smoke jobs from the Gadi login node:

```bash
cd /home/572/aa1215/workspace/repos/ReflectSteer
git submodule update --init --recursive
RELAY=/scratch/qw14/aa1215/ReflectSteer/wiki-relay-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RELAY"
qsub -v WIKI_RELAY_DIR="$RELAY" run_wiki_worker.pbs
```

Wait until `outs/wiki_worker.<job-id>.log` says `Wikipedia worker ready`.
Then, in the same shell (so `RELAY` is still set):

```bash
qsub -v WIKI_RELAY_DIR="$RELAY",HOTPOT_AGENT=react,HOTPOT_STRATEGY=base,HOTPOT_LIMIT=2,HOTPOT_TRIALS=2 run_hotpotqa.pbs
qsub -v WIKI_RELAY_DIR="$RELAY",HOTPOT_AGENT=react,HOTPOT_STRATEGY=reflexion,HOTPOT_LIMIT=2,HOTPOT_TRIALS=2 run_hotpotqa.pbs
```

The worker serves both jobs and exits after 2.9 hours. Cancel it with `qdel`
when both GPU jobs finish. If it expires before a GPU job starts, submit a new
worker with a fresh directory and resubmit the sample. After a forced worker
termination, use a fresh directory rather than reusing its lock file.

The worker invokes the same LangChain Wikipedia docstore as upstream. Full page
documents or suggestion strings are returned; the original GPU-side
`DocstoreExplorer` performs paragraph selection and stateful `Lookup` locally.
Requests/responses are retained in the relay directory, including timestamps,
queries, page text and worker errors. Transport failures abort the GPU run rather
than becoming misleading search observations. No gold paragraphs are supplied.
The relay ping checks worker availability before loading Qwen; successful Search
observations are still needed to confirm actual Wikipedia connectivity.

The Qwen wrapper removes a repeated matching `Thought N:` or `Action N:` prefix
from its completion. Results retain raw `text` and normalized `agent_text`.

The worker's external `wiki_http.py` layer uses HTTPS, an explicit User-Agent,
15-second request timeouts, and up to three attempts for network errors, invalid
JSON, rate limits and temporary server errors. It honors numeric `Retry-After`
headers within a 90-second budget per search. Permanent HTTP failures abort
without retry. API parameters and page/suggestion handling stay with the original
Wikipedia client. `$RELAY/http.jsonl` records HTTP status, URL, content type and a
short response excerpt on failures. The GPU relay timeout defaults to 600 seconds
to allow two jobs to share the serial worker. Persistent blocks still require
diagnosis; retries do not bypass them.

`--agent react` uses the original Wikipedia docstore and Search/Lookup behavior,
six steps per attempt, and the original 3896-token halt rule measured with
`tiktoken` p50k_base. It never substitutes distractor passages. Run it where the
GPU process can access Wikipedia, or configure network access before using it on
NCI. Both the CLI and PBS default to ReAct. A live Wikipedia run also reflects
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
- `reflexion_adapter.py` provides modern LangChain bindings outside the submodule.
  At load time it removes legacy imports and replaces eager OpenAI defaults in
  memory so runners can inject Qwen. Upstream files remain unchanged on disk.
- The external adapter wraps malformed-action parsing to return `(None, None)`
  instead of crashing during tuple unpacking. Original invalid-action feedback
  is retained. This compatibility fix is not an edit to the upstream repository.
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
