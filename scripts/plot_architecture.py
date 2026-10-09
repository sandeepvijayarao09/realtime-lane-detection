#!/usr/bin/env python3
"""
Draw assets/architecture.png from a real forward pass.

Every tensor shape in the figure is read from forward hooks on LaneNet, so the
diagram always matches the code.

    python scripts/plot_architecture.py --backbone efficientnet
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import create_lanenet  # noqa: E402


def collect_shapes(backbone: str, height: int, width: int):
    model = create_lanenet(backbone=backbone, pretrained=False).eval()
    names = ["enc0", "enc1", "enc2", "enc3", "enc4",
             "decoder4", "decoder3", "decoder2", "decoder1", "seg_head", "emb_head"]
    shapes = {}
    hooks = [getattr(model, n).register_forward_hook(
        lambda m, i, o, n=n: shapes.__setitem__(n, tuple(o.shape[1:]))) for n in names]
    with torch.no_grad():
        out = model(torch.zeros(1, 3, height, width))
    for h in hooks:
        h.remove()
    shapes["seg"] = tuple(out["seg"].shape[1:])
    shapes["emb"] = tuple(out["emb"].shape[1:])
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return shapes, params


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "mobilenet"])
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--output", default=str(ROOT / "assets" / "architecture.png"))
    args = parser.parse_args()

    s, params = collect_shapes(args.backbone, args.height, args.width)
    fmt = lambda t: f"{t[0]} × {t[1]} × {t[2]}"  # noqa: E731

    enc = [("Input", (3, args.height, args.width))] + [
        (f"Encoder stage {i}  (1/{2 ** (i + 1)})", s[f"enc{i}"]) for i in range(5)]
    dec = [(f"Decoder  (1/{2 ** (4 - i)})", s[f"decoder{4 - i}"]) for i in range(4)]

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.6, 7.3)
    ax.axis("off")

    enc_color, dec_color, head_color = "#dbe9f6", "#e3f1e0", "#fbe5d6"

    def box(x, y, w, title, shape, color):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.78, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color, ec="#4a4a4a", lw=1))
        ax.text(x + w / 2, y + 0.52, title, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, y + 0.22, fmt(shape), ha="center", va="center", fontsize=9,
                family="monospace", color="#333")

    ys = [6.2 - i * 1.15 for i in range(6)]
    for (title, shape), y in zip(enc, ys):
        box(0.3, y, 3.4, title, shape, enc_color if title != "Input" else "#eeeeee")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.annotate("", (2.0, y1 + 0.78), (2.0, y0), arrowprops=dict(arrowstyle="->", color="#555"))

    dys = [ys[4], ys[3], ys[2], ys[1]]
    for (title, shape), y in zip(dec, dys):
        box(4.3, y, 3.0, title, shape, dec_color)
    # bottleneck -> first decoder, decoder chain upwards
    ax.annotate("", (4.3, dys[0] + 0.39), (3.7, ys[5] + 0.39),
                arrowprops=dict(arrowstyle="->", color="#555", connectionstyle="arc3,rad=0.25"))
    for y0, y1 in zip(dys[:-1], dys[1:]):
        ax.annotate("", (5.8, y1), (5.8, y0 + 0.78), arrowprops=dict(arrowstyle="->", color="#555"))
    # skip connections
    for i, y in enumerate(dys):
        ax.annotate("", (4.3, y + 0.39), (3.7, ys[4 - i] + 0.39),
                    arrowprops=dict(arrowstyle="->", color="#2b7bb9", ls="--"))

    box(7.9, ys[1] + 0.6, 2.8, "Segmentation logits", s["seg"], head_color)
    box(7.9, ys[1] - 0.6, 2.8, "Instance embedding", s["emb"], head_color)
    for yy in (ys[1] + 0.99, ys[1] - 0.21):
        ax.annotate("", (7.9, yy), (7.3, dys[3] + 0.39), arrowprops=dict(arrowstyle="->", color="#555"))

    ax.text(9.3, ys[3] + 0.3, "Heads run at 1/2 resolution,\nthen upsample to the input size.",
            ha="center", fontsize=8.5, color="#555")
    ax.text(9.3, ys[4] + 0.1, "- - -  skip connection", ha="center", fontsize=8.5, color="#2b7bb9")

    backbone = "EfficientNet-B0" if args.backbone == "efficientnet" else "MobileNetV2"
    ax.text(5.5, 7.15, f"LaneNet with {backbone} encoder  ·  {params:.2f} M parameters",
            ha="center", fontsize=12, weight="bold")
    ax.text(5.5, -0.45, "Shapes are channels × height × width, captured with forward hooks "
            "by scripts/plot_architecture.py", ha="center", fontsize=8, color="#777")

    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
