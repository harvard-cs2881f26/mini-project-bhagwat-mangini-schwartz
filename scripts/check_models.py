#!/usr/bin/env python
"""Which of the paper's models does each API still serve? Free: only lists models, sends no prompts.

    python scripts/check_models.py

Reads keys from .env. Providers without a key are reported as "no key" and skipped.
"""
from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from asimov.models import PAPER_MODELS  # noqa: E402

KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def listing(provider: str) -> list[str]:
    if provider == "anthropic":
        from asimov.models.anthropic_client import list_model_ids
    elif provider == "openai":
        from asimov.models.openai_client import list_model_ids
    else:
        from asimov.models.gemini_client import list_model_ids
    return list_model_ids()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Check exact model IDs from this YAML instead of the Fig. 5 stand-ins")
    args = parser.parse_args()
    if args.config:
        import yaml
        specs = yaml.safe_load(Path(args.config).read_text())["models"]
        models = {s["name"]: (s["provider"], s["model"]) for s in specs}
    else:
        models = PAPER_MODELS
    load_dotenv(ROOT / ".env")
    available: dict[str, list[str] | str] = {}
    for provider, var in KEY_VARS.items():
        if not os.environ.get(var):
            available[provider] = "no key"
            continue
        try:
            available[provider] = listing(provider)
        except Exception as e:  # report and keep going
            available[provider] = f"error: {type(e).__name__}: {e}"
    print("Listing only: a listed model may still reject inference for this account.")
    print(f"{'Configured model':24s} {'API model ID':22s} status")
    for name, (provider, model) in models.items():
        ids = available[provider]
        if isinstance(ids, str):
            status = ids
        else:
            # Anthropic lists dated IDs (claude-sonnet-4-20250514); aliases resolve to them.
            stem = model.removesuffix("-0")
            match = [i for i in ids if i == model or i.startswith(stem + "-2")]
            status = "AVAILABLE" + (f" ({match[0]})" if match and match[0] != model else "") if match else "NOT LISTED"
        print(f"{name:24s} {model:22s} {status}")


if __name__ == "__main__":
    main()
