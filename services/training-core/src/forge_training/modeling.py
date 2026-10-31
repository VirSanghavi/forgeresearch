"""Only approved local/Hub weights; remote model code and pickle weights are disabled."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from .artifacts import directory_sha256
from .config import ExperimentConfig

PROMPT_VERSION = "math_completion_v1"
PROMPT_INSTRUCTION = "Solve the problem. End with a final line in the form #### <number>."


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best effort only. Different devices/kernels may still produce different results.
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_device(config: ExperimentConfig) -> None:
    import torch

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if config.device == "mps" and config.precision == "bfloat16":
        if not torch.backends.mps.is_macos_or_newer(14, 0):
            raise ValueError("MPS bfloat16 requires macOS 14 or newer")
    if config.device == "cuda" and config.precision == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16")


def configure_device_memory(config: ExperimentConfig) -> None:
    """Bound this worker's PyTorch GPU allocator before any model allocation."""
    import torch

    if config.device_memory_limit_mb is None:
        return
    requested = config.device_memory_limit_mb * 1024 * 1024
    if config.device == "mps":
        capacity = torch.mps.recommended_max_memory()
        set_fraction = torch.mps.set_per_process_memory_fraction
    elif config.device == "cuda":
        capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        set_fraction = torch.cuda.set_per_process_memory_fraction
    else:
        raise ValueError("device_memory_limit_mb requires an MPS or CUDA device")
    if requested > capacity:
        raise ValueError("Device memory limit exceeds the device's available allocator capacity")
    set_fraction(requested / capacity)


def load_base(config: ExperimentConfig):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    validate_device(config)
    configure_device_memory(config)
    if config.model.source == "local":
        if directory_sha256(Path(config.model.model_id)) != config.model.revision:
            raise ValueError("Local base-model checksum mismatch")
    kwargs = {"trust_remote_code": False, "local_files_only": config.model.source == "local"}
    if config.model.source == "hub":
        kwargs["revision"] = config.model.revision
    architecture = AutoConfig.from_pretrained(config.model.model_id, **kwargs)
    # Count parameters on the meta device before allocating/downloading real weights.
    with init_empty_weights():
        skeleton = AutoModelForCausalLM.from_config(architecture, trust_remote_code=False)
    if sum(parameter.numel() for parameter in skeleton.parameters()) > config.max_model_parameters:
        raise ValueError("Model exceeds configured parameter limit before weight loading")
    del skeleton
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, **kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("The approved tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        **kwargs,
        use_safetensors=True,
        torch_dtype=getattr(torch, config.precision),
        attn_implementation=config.attention_implementation,
        # Materialize weights on the accelerator: staging a CPU copy first briefly holds the
        # model twice (Qwen2.5-0.5B float32 on MPS: about 3.5 GB peak instead of 1.9 GB).
        **({} if config.device == "cpu" else {"device_map": {"": config.device}}),
    )
    if sum(parameter.numel() for parameter in model.parameters()) > config.max_model_parameters:
        raise ValueError("Model exceeds configured parameter limit")
    model.to(config.device)
    return model, tokenizer


def prompt_for(tokenizer, question: str, instruction: str = PROMPT_INSTRUCTION) -> str:
    content = f"{instruction}\n\n{question}"
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Question: {content}\nAnswer:"


def parameter_sha256(model) -> str:
    """Hash trainable tensor bytes, including names/shapes, before and after optimization."""
    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(str(parameter.dtype).encode())
            tensor = parameter.detach().contiguous().cpu().view(torch.uint8)
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
