import os, sys

from evolix.config.config import Config
from evolix.data.tokenizer import Tokenizer
from evolix.engine.generator import Generator
from evolix.utils.checkpoint import CheckpointManager


def infer(prompt: str, max_new_tokens: int = 256, temperature: float = 0.8, top_k: int = 0, top_p: float = 0.9) -> str:
    cfg = Config()
    cfg.hf_token = os.environ.get("HF_TOKEN", None) if not cfg.hf_token else cfg.hf_token
    if not cfg.hf_token:
        raise ValueError("Hugging Face token is not set. Please set the 'hf_token' in the configuration.")

    tokenizer = Tokenizer(f"src/evolix/data/{cfg.spm_prefix}.model")
    checkpoint_manager = CheckpointManager(cfg, require_write=False)
    prompt = prompt if prompt else "Hello"
    generator = Generator(cfg, checkpoint_manager, tokenizer).load()
    text = generator.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    print(text)
    return text


if __name__ == "__main__":
    infer(sys.argv[1] if len(sys.argv) > 1 else "Hello,")
