"""Smoke test: construct + forward pass on every registered OWL model.

OWLC and OWLT use the DLA-34 backbone (pretrained=False here so the
test does not depend on torchvision ImageNet downloads).

OWLD_{S,B,L,H} need DINOv3 backbone weights under weights/. The script
looks them up by the filenames hard-coded on each class
(_DEFAULT_WEIGHTS_FILENAME). If a weight file is missing, that OWL-D
variant is skipped (reported as SKIPPED rather than FAIL).

Exit code:
  0 = every available model PASSED (missing OWL-D weights count as
      SKIPPED, not FAIL)
  1 = at least one model failed to construct or forward
"""

import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# OWL-D weights live under <repo>/weights/ by default; see INSTALL.md.
WEIGHTS_DIR = REPO_ROOT / "weights"


def smoke_one(name: str, model: torch.nn.Module, x: torch.Tensor) -> tuple[bool, str]:
    try:
        with torch.no_grad():
            out = model(x)
        if isinstance(out, (list, tuple)):
            shapes = [tuple(o.shape) for o in out if hasattr(o, "shape")]
            return True, f"out shapes={shapes}"
        return True, f"out shape={tuple(out.shape)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def main() -> int:
    from animaloc.models import MODELS

    KWARGS = {
        # DLA-34 backbone (skip torchvision pretrained to avoid network deps).
        "OWLC": dict(num_layers=34, pretrained=False, down_ratio=2, head_conv=64),
        # OWLT: DLA-34 + Swin (note kwarg name: pretrained_cnn, not pretrained).
        "OWLT": dict(num_layers=34, pretrained_cnn=False, down_ratio=2, head_conv=64),
        # OWLD_*: load DINOv3 weights from disk if present.
        "OWLD_S": dict(down_ratio=2, freeze_backbone=True),
        "OWLD_B": dict(down_ratio=2, freeze_backbone=True),
        "OWLD_L": dict(down_ratio=2, freeze_backbone=True),
        "OWLD_H": dict(down_ratio=2, freeze_backbone=True),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}; weights/={WEIGHTS_DIR} (exists={WEIGHTS_DIR.exists()})")

    x = torch.randn(1, 3, 512, 512, device=device)

    rows = []
    fail = 0
    skip = 0
    for name, kwargs in KWARGS.items():
        print(f"\n[{name}] constructing with {kwargs}")
        try:
            cls = MODELS[name]
            model = cls(**kwargs).to(device).eval()
        except FileNotFoundError as e:
            # Missing DINOv3 weights -- skip, do not fail.
            print(f"[{name}] SKIPPED: {e}")
            rows.append((name, "SKIPPED", f"weights not found: {e}"))
            skip += 1
            continue
        except Exception as e:
            print(f"[{name}] FAIL: construction: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            rows.append((name, "CONSTRUCT_FAIL", str(e)))
            fail += 1
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{name}] params={n_params:,}, forwarding 1x3x512x512 ...")
        ok, msg = smoke_one(name, model, x)
        if ok:
            print(f"[{name}] PASS  {msg}")
            rows.append((name, "PASS", msg))
        else:
            print(f"[{name}] FAIL  {msg}", file=sys.stderr)
            rows.append((name, "FORWARD_FAIL", msg.split("\n", 1)[0]))
            fail += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_pass = len(rows) - fail - skip
    print("\n=== Forward-pass smoke summary ===")
    for name, status, msg in rows:
        line = msg if len(msg) < 80 else msg[:77] + "..."
        print(f"  {name:8s}  {status:14s}  {line}")
    print(f"\n{n_pass}/{len(rows)} models passed ({skip} skipped); exit={int(fail > 0)}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
