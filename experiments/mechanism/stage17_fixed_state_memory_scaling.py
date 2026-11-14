"""Measure shadow-cache scaling for a fixed-state Mamba and a Transformer."""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MambaForCausalLM

from stage5_tab_context_validation import load_candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--rwkv", default="RWKV/rwkv-4-169m-pile")
    parser.add_argument("--transformer", default="openai-community/gpt2")
    parser.add_argument("--lengths", default="32,64,128,256,512")
    parser.add_argument("--output", default="results/stage17_fixed_state_memory_scaling.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    if hasattr(value, "to_legacy_cache"):
        return tensor_bytes(value.to_legacy_cache())
    return 0


def real_text() -> str:
    occurrences = load_candidates(80)
    snippets = []
    seen = set()
    for rows in occurrences.values():
        for row in rows:
            text = row["redacted_context"]
            if text not in seen:
                snippets.append(text)
                seen.add(text)
            if len(snippets) == 12:
                return " The public legal record states: ".join(snippets)
    return " The public legal record states: ".join(snippets)


def ids_at_length(tokenizer, text, length, device):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError("tokenizer produced no tokens")
    repeated = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
    return torch.tensor([repeated], dtype=torch.long, device=device)


@torch.inference_mode()
def measure_mamba(model_name, lengths, text, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    rows = []
    for length in lengths:
        output = model(input_ids=ids_at_length(tokenizer, text, length, device), use_cache=True, return_dict=True)
        cache = output.cache_params
        cache_bytes = tensor_bytes(cache.ssm_states) + tensor_bytes(cache.conv_states)
        rows.append({"context_tokens": length, "one_cache_bytes": cache_bytes, "shadow_increment_bytes": cache_bytes})
        del output
    config = {
        "layers": model.config.num_hidden_layers,
        "state_size": model.config.state_size,
        "conv_kernel": model.config.conv_kernel,
        "intermediate_size": model.config.intermediate_size,
    }
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows, config


@torch.inference_mode()
def measure_transformer(model_name, lengths, text, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    rows = []
    for length in lengths:
        output = model(input_ids=ids_at_length(tokenizer, text, length, device), use_cache=True, return_dict=True)
        cache_bytes = tensor_bytes(output.past_key_values)
        rows.append({"context_tokens": length, "one_cache_bytes": cache_bytes, "shadow_increment_bytes": cache_bytes})
        del output
    config = {
        "layers": getattr(model.config, "n_layer", model.config.num_hidden_layers),
        "heads": getattr(model.config, "n_head", model.config.num_attention_heads),
        "hidden_size": getattr(model.config, "n_embd", model.config.hidden_size),
    }
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows, config


@torch.inference_mode()
def measure_rwkv(model_name, lengths, text, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    rows = []
    for length in lengths:
        output = model(input_ids=ids_at_length(tokenizer, text, length, device), use_cache=True, return_dict=True)
        cache_bytes = tensor_bytes(output.state)
        rows.append({"context_tokens": length, "one_cache_bytes": cache_bytes, "shadow_increment_bytes": cache_bytes})
        del output
    config = {
        "layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
    }
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows, config


def fit_slope(rows):
    first, last = rows[0], rows[-1]
    return (last["one_cache_bytes"] - first["one_cache_bytes"]) / (
        last["context_tokens"] - first["context_tokens"]
    )


def main():
    args = parse_args()
    lengths = sorted({int(item) for item in args.lengths.split(",")})
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    text = real_text()
    mamba_rows, mamba_config = measure_mamba(args.mamba, lengths, text, args.device, dtype)
    rwkv_rows, rwkv_config = measure_rwkv(args.rwkv, lengths, text, args.device, dtype)
    transformer_rows, transformer_config = measure_transformer(
        args.transformer, lengths, text, args.device, dtype
    )
    output = {
        "status": "stage_17_fixed_state_vs_transformer_shadow_memory",
        "configuration": {**vars(args), "lengths": lengths, "dtype": str(dtype)},
        "data_protocol": "token lengths drawn from real TAB legal text; no excerpts emitted",
        "mamba": {"config": mamba_config, "rows": mamba_rows, "fitted_bytes_per_context_token": fit_slope(mamba_rows)},
        "rwkv": {"config": rwkv_config, "rows": rwkv_rows, "fitted_bytes_per_context_token": fit_slope(rwkv_rows)},
        "transformer": {
            "config": transformer_config,
            "rows": transformer_rows,
            "fitted_bytes_per_context_token": fit_slope(transformer_rows),
        },
        "claim": "For batch size one, an additional Mamba shadow cache is context-length invariant; an additional Transformer KV cache grows linearly in context length.",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, output, newline=True)
    print(json.dumps({
        "output": str(destination),
        "mamba_bytes": [row["one_cache_bytes"] for row in mamba_rows],
        "rwkv_bytes": [row["one_cache_bytes"] for row in rwkv_rows],
        "transformer_bytes": [row["one_cache_bytes"] for row in transformer_rows],
        "mamba_bytes_per_token": output["mamba"]["fitted_bytes_per_context_token"],
        "rwkv_bytes_per_token": output["rwkv"]["fitted_bytes_per_context_token"],
        "transformer_bytes_per_token": output["transformer"]["fitted_bytes_per_context_token"],
    }, indent=2))


if __name__ == "__main__":
    main()
