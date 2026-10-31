"""Work out how Forge trains a Hugging Face model, without downloading its weights.

The architecture is built on the meta device from the model's config at a pinned commit, so the
LoRA targets come from the real module names and even a 20B model costs almost no memory. Remote
model code and pickle weights are refused. Run as a module so the API never has to import torch:

    python -m services.core.model_inspect <owner/name> [<revision>]
"""

import json
import re
import sys

# Attention projections LoRA should adapt, by the names architectures give them. Query and value
# projections where they are separate, the combined projection where they are not.
TARGET_NAMES = {
    "q_proj",
    "v_proj",
    "q",
    "v",
    "query",
    "value",
    "query_key_value",
    "qkv",
    "qkv_proj",
    "c_attn",
    "Wqkv",
    "att_proj",
    "in_proj_qkv",
    "q_lin",
    "v_lin",
    "wq",
    "wv",
    "to_q",
    "to_v",
}
OUTPUT_NAMES = {"out_proj", "o_proj", "resid_dropout", "c_proj", "output", "out_lin", "wo", "to_out"}
ATTENTION = re.compile(r"(^|[._])(self_)?(attn|attention|attentions)($|[._])", re.IGNORECASE)
NON_TEXT = re.compile(r"(^|\.)(vision|visual|image|audio|multi_modal)", re.IGNORECASE)
THINKING = "enable_thinking"


class Unsupported(ValueError):
    """This model cannot be trained here; the message says why."""


def label_for(repo):
    name = repo.split("/")[-1]
    return re.sub(r"[-_](instruct|chat|hf|base|it)$", "", name, flags=re.IGNORECASE).replace("-", " ")


def format_parameters(count):
    if count >= 999_500_000:
        return f"{count / 1e9:.3g}B"
    return f"{count / 1e6:.0f}M"


def lora_targets(named_modules):
    """Pick LoRA target names from (module path, module class name) pairs."""
    found = set()
    for path, kind in named_modules:
        leaf = path.rsplit(".", 1)[-1]
        if kind not in ("Linear", "Conv1D") or NON_TEXT.search(path):
            continue
        if ATTENTION.search(path) and "." in path:
            found.add(leaf)
    targets = sorted(found & TARGET_NAMES)
    if not targets:
        targets = sorted(found - OUTPUT_NAMES)
    if not targets:
        raise Unsupported("No attention layers that LoRA can adapt were found in this architecture")
    return targets


def _template_kwargs(template):
    # The documented non-thinking mode: the "final answer" protocols read a direct answer.
    return {THINKING: False} if isinstance(template, str) and THINKING in template else {}


def inspect(repo, revision=None):
    import torch
    from accelerate import init_empty_weights
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from forge_training.modeling import prompt_for

    try:
        info = HfApi().model_info(
            repo, revision=revision, expand=["sha", "gated", "safetensors", "cardData", "siblings", "config"]
        )
    except RepositoryNotFoundError as error:
        raise Unsupported(f"{repo} was not found on Hugging Face (or needs access)") from error
    sha = info.sha
    files = [s.rfilename for s in info.siblings or []]
    if not any(name.endswith(".safetensors") for name in files):
        raise Unsupported("This model has no safetensors weights, and pickle weights are refused")
    try:
        config = AutoConfig.from_pretrained(repo, revision=sha, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=sha, trust_remote_code=False)
    except (GatedRepoError, OSError, ValueError, KeyError, ImportError) as error:
        public = (info.config or {}) if info.gated else {}
        if not public.get("model_type"):
            raise Unsupported(f"This model needs code or files Forge will not run: {error}") from error
        # Gated without access yet: the architecture's own default config has the same module
        # names, and loading with HF_TOKEN later proves the tokenizer.
        config, tokenizer = AutoConfig.for_model(public["model_type"]), None
    if tokenizer is not None and tokenizer.eos_token_id is None:
        raise Unsupported("Its tokenizer defines no end-of-text token")
    try:
        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    except (ValueError, KeyError, ImportError, TypeError) as error:
        raise Unsupported(f"Not a text-generation architecture Forge can load: {error}") from error
    # Subclasses count: Falcon's attention uses its own FalconLinear.
    kinds = (
        (path, "Linear" if isinstance(module, torch.nn.Linear) else type(module).__name__)
        for path, module in skeleton.named_modules()
    )
    targets = lora_targets(kinds)
    built = sum(p.numel() for p in skeleton.parameters())
    count = (info.safetensors.total if info.safetensors else 0) or built
    template = (
        tokenizer.chat_template
        if tokenizer is not None
        else ((info.config or {}).get("tokenizer_config") or {}).get("chat_template")
    )
    kwargs = _template_kwargs(template)
    # Trust bfloat16 only where the weights ship in it. Forcing a half-precision type on other
    # models is not safe: Pythia returns NaN in float16 and a much worse loss in bfloat16.
    stored = info.safetensors.parameters if info.safetensors else {}
    dtype = "bfloat16" if stored and max(stored, key=stored.get) == "BF16" else "float32"
    if tokenizer is not None:
        if kwargs:
            tokenizer.apply_chat_template = _with(tokenizer.apply_chat_template, kwargs)
        try:
            prompt_for(tokenizer, "1 + 1?", "Answer.")
        except Exception as error:  # Templates are arbitrary Jinja; any failure means unusable.
            raise Unsupported(f"Its chat template cannot render a question: {error}") from error
    del skeleton
    card = info.card_data
    return {
        "id": repo,
        "revision": sha,
        "label": label_for(repo),
        "family": repo.split("/")[0],
        "architecture": config.model_type,
        "parameters": format_parameters(count),
        "parameter_count": count,
        "dtype": dtype,
        "lora_targets": targets,
        "chat_template_kwargs": kwargs,
        "max_new_tokens": 512 if kwargs else 256,
        "chat_template": bool(template),
        "license": (card.license if card else None) or "see model card",
        "gated": bool(info.gated),
    }


def _with(function, kwargs):
    import functools

    return functools.partial(function, **kwargs)


if __name__ == "__main__":
    try:
        print(json.dumps(inspect(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)))
    except Unsupported as error:
        print(json.dumps({"error": str(error)}))
