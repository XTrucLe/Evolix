import os, sys

from evolix.config.config import Config
from evolix.data.tokenizer import Tokenizer
from evolix.engine.generator import Generator
from evolix.utils.checkpoint import CheckpointManager


def infer(max_new_tokens: int = 2048, temperature: float = 0.8, top_k: int = 0, top_p: float = 0.9) -> str:
    cfg = Config()
    cfg.hf_token = os.environ.get("HF_TOKEN", None) if not cfg.hf_token else cfg.hf_token
    if not cfg.hf_token:
        raise ValueError("Hugging Face token is not set. Please set the 'hf_token' in the configuration.")

    tokenizer = Tokenizer(f"evolix/artifacts/tokenizers/{cfg.spm_prefix}_vocab.json")
    checkpoint_manager = CheckpointManager(cfg)
    generator = Generator(cfg, checkpoint_manager, tokenizer).load()

    while True:
        try:
            prompt = input("User: ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        if prompt.lower() in ["q", "quit", "exit", "[q]", "[quit]", "[exit]"]:
            print("\nExiting...")
            break
        generated_text = generator.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        print(f"{cfg.spm_prefix.capitalize()}:\n{generated_text}\n")


if __name__ == "__main__":
    infer()
