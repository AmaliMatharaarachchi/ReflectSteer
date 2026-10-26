"""Reusable offline Qwen chat inference; imports GPU dependencies only on load."""
import json
import os
from pathlib import Path


class QwenLM:
    def __init__(self, model_dir, device='cuda', max_input_tokens=3072,
                 max_new_tokens=128):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        model_dir = Path(model_dir).expanduser().resolve()
        if not model_dir.is_dir():
            raise ValueError(f'Local model directory does not exist: {model_dir}')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; run inside an NCI GPU allocation')
        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, device_map=device,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32).eval()
        self.metadata = {'model_dir': str(model_dir), 'device': device,
                         'dtype': str(self.model.dtype),
                         'model_type': self.model.config.model_type,
                         'revision': getattr(self.model.config, '_commit_hash', None)}
        manifest = model_dir / 'checkpoint_manifest.json'
        if manifest.is_file():
            self.metadata['checkpoint_manifest'] = json.loads(manifest.read_text(encoding='utf-8'))

    def count_tokens(self, messages):
        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True))

    def generate(self, messages, temperature=0, top_p=0.9, max_new_tokens=None, stop=None):
        if temperature < 0 or not 0 < top_p <= 1:
            raise ValueError('Invalid sampling parameters')
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        n_input = inputs['input_ids'].shape[1]
        if n_input > self.max_input_tokens:
            raise ValueError(f'Prompt has {n_input} tokens; limit is {self.max_input_tokens}')
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        context_limit = getattr(self.model.config, 'max_position_embeddings', None)
        if context_limit and n_input + max_new_tokens > context_limit:
            raise ValueError('Prompt plus generation exceeds model context window')
        inputs = inputs.to(self.model.device)
        options = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                       pad_token_id=self.tokenizer.eos_token_id)
        if temperature > 0:
            options.update(temperature=temperature, top_p=top_p)
        if stop:
            options.update(stop_strings=stop, tokenizer=self.tokenizer)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        ids = output[0, n_input:]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        if stop:
            text = text[:min([text.index(s) for s in stop if s in text] or [len(text)])]
        return {'text': text,
                'input_tokens': n_input, 'output_tokens': len(ids)}
