"""Check that TransformerLens (no weight processing) reproduces the HF model.

Compares per-token log-probabilities and clean_loss on the same validation
texts. Usage: pixi run python -m zoo.check_tl --config configs/check_tl.yaml
"""

from __future__ import annotations

import argparse
import json

import torch

from zoo.config import load_config
from zoo.data import load_split
from zoo.model import load_base_model, load_tl_bridge, load_tokenizer
from zoo.readouts import clean_loss
from zoo.runs import Run, make_run_id
from zoo.seeding import set_seed


class _LogitsOut:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class BridgeAsHF(torch.nn.Module):
    """Expose a TransformerBridge with the HF call signature used by the readouts."""

    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge

    def forward(self, input_ids: torch.Tensor) -> _LogitsOut:
        return _LogitsOut(self.bridge(input_ids, return_type="logits"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run(make_run_id("check_tl"), "check_tl", cfg, args.config, seed=cfg["seed"]).start()
    try:
        set_seed(cfg["seed"])
        device = torch.device(cfg["device"])
        tok = load_tokenizer(cfg)
        c = cfg["check"]
        texts = load_split(cfg, c["split"])["text"][c["start"]:c["start"] + c["n_texts"]]
        ids = [tok(t)["input_ids"][: c["max_tokens"] - 1] for t in texts]
        hf = load_base_model(cfg, device)
        tl = BridgeAsHF(load_tl_bridge(cfg, device)).eval()
        with torch.no_grad():
            x = torch.tensor([[tok.bos_token_id] + ids[0]], device=device)
            max_abs_logit_diff = float((hf(input_ids=x).logits - tl(x).logits).abs().max())
        hf_loss = clean_loss(hf, ids, tok.bos_token_id, cfg["readouts"]["max_batch_tokens"], device)
        tl_loss = clean_loss(tl, ids, tok.bos_token_id, cfg["readouts"]["max_batch_tokens"], device)
        res = {"hf_clean_loss": hf_loss, "tl_clean_loss": tl_loss, "abs_diff": abs(hf_loss - tl_loss),
               "max_abs_logit_diff_first_text": max_abs_logit_diff,
               "tolerance": c["tolerance"], "pass": abs(hf_loss - tl_loss) < c["tolerance"],
               "n_texts": len(ids), "n_params_hf": sum(p.numel() for p in hf.parameters())}
        (run.dir / "result.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        run.finish("ok" if res["pass"] else "failed", reason="" if res["pass"] else "TL loss differs from HF")
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


if __name__ == "__main__":
    main()
