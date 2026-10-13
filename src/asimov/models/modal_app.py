"""Modal app serving the local model clients on a GPU (params `backend: modal`; models/remote.py).

Deploy (or redeploy after changing src/asimov, e.g. the wording file), from the repo root:
    PYTHONPATH=src modal deploy -m asimov.models.modal_app
Stop (not needed: containers scale to zero after SCALEDOWN_S idle seconds):
    modal app stop asimov-local-models

Each class is parametrized by the config's params, so one container holds one model. In
@modal.enter it builds the ordinary client (backend local) with this repo's code (src/asimov is
copied into the image), on CUDA; weights download once to a Modal Volume at the pinned revision.
Public models only: no secrets are sent. Only imported by `modal deploy`, never by the runner or CI.
No `from __future__ import annotations`: Modal reads the class parameters' types at deploy time.
"""

import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import modal

from asimov.models.remote import APP, serve_complete, serve_decide

GPU = "L40S"  # as Kev-4B's deploy (part2_kev), so their latencies compare like for like
SCALEDOWN_S = 300
TIMEOUT_S = 600  # per call
STARTUP_S = 1200  # container start: the first one per model downloads ~9 GB of weights to the Volume
CACHE = "/cache"
SRC = Path(__file__).resolve().parents[2]  # src/
# Qwen3.5's linear-attention layers use these Triton kernels when installed; otherwise transformers
# falls back to a much slower PyTorch reference (as on the laptop). Recorded per reply as `kernels`.
FLA = "flash-linear-attention==0.5.2"

app = modal.App(APP)
weights = modal.Volume.from_name("asimov-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(  # requirements-local.txt versions; pyyaml/pydantic for asimov's own imports
        "torch==2.14.1", "transformers==5.19.0", "accelerate==1.15.0", "sentencepiece==0.2.2", "pyyaml", "pydantic", FLA
    )  # fmt: skip
    .env({"HF_HOME": CACHE, "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_dir(SRC / "asimov", "/root/asimov", ignore=["**/__pycache__"])
)
SETTINGS: dict[str, Any] = dict(
    image=image, gpu=GPU, volumes={CACHE: weights}, scaledown_window=SCALEDOWN_S, timeout=TIMEOUT_S,
    startup_timeout=STARTUP_S, max_containers=1,
)  # fmt: skip


def where(client: Any, load_s: float) -> dict:
    """Where replies are made, saved with each one (raw["remote"])."""
    import torch

    kernels = "fla" if find_spec("fla") else "torch reference"
    gpu = torch.cuda.get_device_name(0)
    return {"backend": "modal", "device": client.device, "gpu": gpu, "kernels": kernels, "load_s": round(load_s, 1)}


@app.cls(**SETTINGS)
class LocalHF:
    repo: str = modal.parameter()
    revision: str = modal.parameter()
    mode: str = modal.parameter()
    template: str = modal.parameter()
    logit_dtype: str = modal.parameter(default="fp32")
    max_new_tokens: int = modal.parameter(default=1024)  # generate mode only (models.local_hf.MAX_NEW_TOKENS)

    @modal.enter()
    def load(self) -> None:
        from asimov.models.local_hf import LocalHFClient

        t0 = time.perf_counter()
        params = {"repo": self.repo, "revision": self.revision, "mode": self.mode, "template": self.template,
                  "logit_dtype": self.logit_dtype, "max_new_tokens": self.max_new_tokens}  # fmt: skip
        self.client = LocalHFClient("modal", params)
        self.info = where(self.client, time.perf_counter() - t0)
        weights.commit()

    @modal.method()
    def decide(self, part: dict) -> dict:
        return serve_decide(self.client, part, self.info)

    @modal.method()
    def complete(self, text: str) -> dict:
        return serve_complete(self.client, text, self.info)


@app.cls(**SETTINGS)
class NLI:
    repo: str = modal.parameter()
    revision: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from asimov.models.nli import NLIClient

        t0 = time.perf_counter()
        self.client = NLIClient("modal", {"repo": self.repo, "revision": self.revision})
        self.info = where(self.client, time.perf_counter() - t0)
        weights.commit()

    @modal.method()
    def decide(self, part: dict) -> dict:
        return serve_decide(self.client, part, self.info)
