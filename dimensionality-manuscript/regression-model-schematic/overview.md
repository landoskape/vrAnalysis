# Neuro-Model Schematic Builder (Python) — Design Spec

A complete specification (with code scaffolding) for a small, composable Python library that generates **vector-quality schematics** for your six CA1 prediction models. Copy this into Cursor and iterate from here.

---

## Overview

This library renders six panels left→right along a conceptual axis **“Spatially constrained → Unconstrained”**, with optional region annotations **External → Internal → Peer**. Each panel is a lightweight **diagrammatic cartoon** made of reusable primitives (track, PF bumps, neuron blocks, decode/encode/gain boxes, arrows, legends, train/test bars).

**Primary goals**
- One function per model panel and a **declarative API** to compose figures.
- **Matplotlib**-only for portability; vector-friendly (PDF/SVG) output.
- Crisp, consistent **visual language**; color-blind-safe palette.
- Deterministic layouts via a shared grid + sizing constants.

**Models (left → right)**
1. External PF  
2. Internal PF (split-half decode)  
3. External PF + gain  
4. Internal PF + gain  
5. High-D internal PF (RBF decode→encode)  
6. RRR (reduced-rank, non-spatial peer)  

Recommended ordering (slightly smoother conceptual gradient): **1 → 3 → 2 → 4 → 5 → 6**.

---

## Installation

```bash
pip install matplotlib numpy
```

*(Optional)* If you want nicer arrows:  
`pip install "matplotlib>=3.8"`

---

## Visual Language

- **Track / position axis**: horizontal line with ticks.  
- **Measured position** \(x(t)\): gold dot on track (external).  
- **Decoded position** \(\hat x(t)\): grey dot, dashed outline (internal).  
- **Place field (PF)**: bell curve(s) above track; target in red, sources in teal.  
- **Neurons**: circles in a grouped block; target (red), sources (teal).  
- **Matrices / maps**: rounded rectangles with labels (e.g., “× g”, “RRR, rank r”).  
- **Arrows**: solid = known/external; dashed = decoded/latent; double = two-stage.  
- **Train/test stripe**: thin bar below panel (left shaded “train”, right unshaded “test”).  
- **Split-half**: source/target population separated by a subtle divider; lock icon on held-out target.  
- **Uncertainty**: faint blur or multiple lit RBF bumps.

**Color palette (color-blind safe)**
- External (true) **gold**: `#C9A227`  
- Internal (decoded) **grey**: `#6B7280` (outline dashed)  
- Source neurons **teal**: `#2CA7A3`  
- Target neurons **red**: `#D1495B`  
- Basis / features **purple**: `#7A5CCE`  
- Neutral lines **charcoal**: `#111827`

---

## Project Structure

```
schematics/
  __init__.py
  config.py
  primitives.py
  layout.py
  panels.py
  figure.py
  examples/
    build_full_figure.py
```

---

## Configuration (`config.py`)

```python
PALETTE = {
    "external": "#C9A227",
    "internal": "#6B7280",
    "source":   "#2CA7A3",
    "target":   "#D1495B",
    "basis":    "#7A5CCE",
    "stroke":   "#111827",
}

SIZES = {
    "panel_w": 3.0,
    "panel_h": 2.2,
    "dpi": 300,
    "lw": 1.5,
    "tick": 0.06,
    "pf_height": 0.25,
}

FONTS = {
    "family": "DejaVu Sans",
    "size": 9,
    "small": 8,
    "title": 10,
}
```

---

## Primitives (`primitives.py`)

Each primitive draws relative to a **panel-local coordinate system** \([0,1] \times [0,1]\).

- `Track(ax, x0=0.1, x1=0.9, y=0.25, ticks=5, color="stroke")`
- `PositionDot(ax, x, y, color, dashed=False, alpha=1.0)`
- `PF(ax, center, width, height, y_base, color, lw)`
- `NeuronBlock(ax, x, y, w, h, n_rows, n_cols, color, label=None, divider=False, locked=False)`
- `Box(ax, x, y, w, h, label, color="stroke", fill=False, rounded=True)`
- `Arrow(ax, x0, y0, x1, y1, style="solid"|"dashed"|"double")`
- `Legend(ax, items=[("External pos.", "external"), ...])`
- `TrainTestBar(ax, x0, x1, y, label_left="train", label_right="test")`
- `IconDial(ax, x, y, label="g")`
- `SVDBox(ax, x, y, w, h, r)`

---

## Layout Helpers (`layout.py`)

- `PanelContext(ax)` — convenience methods:
  - `.anchor(name)` → returns (x,y)
  - `.text_center(x, y, s, size="small")`
- `grid_row(n_panels, panel_size)` — returns `(fig, axes)`.

---

## Panel Contracts (`panels.py`)

Each panel function:
- Draws the cartoon (top ~70%).
- Adds title + subtitle.
- Adds one-line equation.

### 1) External PF

**Equation**: \(\hat y(t)=f(x(t))\)

```python
def panel_external_pf(ax, ctx, opt):
    Track(ax)
    PositionDot(ax, x=0.6, y=0.25, color=PALETTE["external"])
    PF(ax, center=0.6, width=0.12, height=0.25, y_base=0.45, color=PALETTE["target"], lw=SIZES["lw"])
    Arrow(ax, 0.6, 0.27, 0.6, 0.45, style="solid")
    ctx.text_center(0.5, 0.95, "External PF (1-D)", size="title")
    ctx.text_center(0.5, 0.89, "Purely spatial; measured position only.")
    ctx.text_center(0.5, 0.07, r"$\hat y(t)=f(x(t))$")
```

### 2) Internal PF (split-half)

**Equation**: \(\hat x=D\,y_{\text{src}};\ \hat y=f(\hat x)\)

```python
def panel_internal_pf(ax, ctx, opt):
    NeuronBlock(ax, x=0.08, y=0.55, w=0.22, h=0.30, n_rows=3, n_cols=5, color=PALETTE["source"], label="sources")
    Arrow(ax, 0.19, 0.55, 0.45, 0.27, style="dashed")
    Track(ax)
    PositionDot(ax, x=0.45, y=0.25, color=PALETTE["internal"], dashed=True)
    PF(ax, center=0.45, width=0.12, height=0.25, y_base=0.45, color=PALETTE["target"], lw=SIZES["lw"])
    Arrow(ax, 0.45, 0.27, 0.45, 0.45, style="dashed")
    NeuronBlock(ax, x=0.78, y=0.55, w=0.14, h=0.20, n_rows=2, n_cols=3, color=PALETTE["target"], label="targets", locked=True)
    Arrow(ax, 0.53, 0.52, 0.78, 0.60, style="solid")
    ctx.text_center(0.5, 0.95, "Internal PF (1-D)", size="title")
    ctx.text_center(0.5, 0.89, "Decode position from peers; PF lookup.")
    ctx.text_center(0.5, 0.07, r"$\hat x=D\,y_{\rm src};\ \hat y=f(\hat x)$")
    TrainTestBar(ax, 0.15, 0.85, y=0.12)
```

### 3) External PF + gain

**Equation**: \(\hat y(t)=g\,f(x(t))\)

```python
def panel_external_pf_gain(ax, ctx, opt):
    Track(ax); PositionDot(ax, x=0.55, y=0.25, color=PALETTE["external"])
    PF(ax, center=0.55, width=0.12, height=0.25, y_base=0.50, color=PALETTE["target"], lw=SIZES["lw"])
    Arrow(ax, 0.55, 0.27, 0.55, 0.50, style="solid")
    Box(ax, x=0.63, y=0.46, w=0.10, h=0.08, label="× g")
    Arrow(ax, 0.73, 0.50, 0.85, 0.58, style="solid")
    NeuronBlock(ax, x=0.10, y=0.70, w=0.18, h=0.16, n_rows=2, n_cols=4, color=PALETTE["source"], label="sources")
    Arrow(ax, 0.28, 0.70, 0.68, 0.50, style="dashed")
    ctx.text_center(0.5, 0.95, "External PF + Gain", size="title")
    ctx.text_center(0.5, 0.89, "Global modulation on spatial prediction.")
    ctx.text_center(0.5, 0.07, r"$\hat y=g\,f(x(t))$")
    TrainTestBar(ax, 0.15, 0.85, y=0.12)
```

### 4) Internal PF + gain

Combine (2) + gain box.

### 5) High-D internal PF (RBF decode→encode)

**Equation**: \(\hat\phi=D\,y_{\rm src};\ \hat y=E\,\hat\phi\)

*(see previous version for full example implementation)*

### 6) RRR (non-spatial peer)

**Equation**: \(\hat Y=W_{\rm RRR}Y_{\rm src},\ \mathrm{rank}(W)=r\)

```python
def panel_rrr(ax, ctx, opt):
    NeuronBlock(ax, x=0.08, y=0.60, w=0.22, h=0.26, n_rows=3, n_cols=5, color=PALETTE["source"], label="sources")
    Box(ax, x=0.40, y=0.60, w=0.18, h=0.12, label="RRR\nrank r")
    SVDBox(ax, x=0.40, y=0.60, w=0.18, h=0.12, r=3)
    Arrow(ax, 0.30, 0.66, 0.40, 0.66, style="solid")
    NeuronBlock(ax, x=0.72, y=0.60, w=0.18, h=0.26, n_rows=3, n_cols=5, color=PALETTE["target"], label="targets")
    Arrow(ax, 0.58, 0.66, 0.72, 0.66, style="solid")
    ctx.text_center(0.5, 0.95, "RRR (Non-spatial)", size="title")
    ctx.text_center(0.5, 0.89, "Peer prediction; no spatial variables.")
    ctx.text_center(0.5, 0.07, r"$\hat Y=W_{\rm RRR}Y_{\rm src},\ \mathrm{rank}(W)=r$")
    TrainTestBar(ax, 0.15, 0.85, y=0.12)
```

---

## Figure Assembly (`figure.py`)

```python
import matplotlib.pyplot as plt
from config import SIZES
from panels import (
    panel_external_pf, panel_external_pf_gain,
    panel_internal_pf, panel_internal_pf_gain,
    panel_rbf_internal, panel_rrr
)

def build_row(order=(1,3,2,4,5,6), figsize=None, dpi=None, annotate_axis=True):
    n = len(order)
    figsize = figsize or (n * SIZES["panel_w"], SIZES["panel_h"])
    dpi = dpi or SIZES["dpi"]
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi, constrained_layout=True)
    if n == 1: axes = [axes]
    mapping = {
        1: panel_external_pf,
        2: panel_internal_pf,
        3: panel_external_pf_gain,
        4: panel_internal_pf_gain,
        5: panel_rbf_internal,
        6: panel_rrr,
    }
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(0,1); ax.set_ylim(0,1)
    for ax, idx in zip(axes, order):
        mapping[idx](ax, ctx=PanelContext(ax), opt={})
    if annotate_axis:
        annotate_concept_axis(fig, axes)
    return fig, axes

def annotate_concept_axis(fig, axes):
    fig.suptitle("Spatially constrained → Unconstrained", y=0.995, fontsize=11)
    ax0 = axes[0]; y = 1.06
    ax0.text(0.08, y, "External", transform=ax0.transAxes, ha="left", va="bottom", fontsize=9)
    axes[len(axes)//2].text(0.5, y, "Internal", transform=axes[len(axes)//2].transAxes, ha="center", va="bottom", fontsize=9)
    axes[-1].text(0.92, y, "Peer", transform=axes[-1].transAxes, ha="right", va="bottom", fontsize=9)

def export(fig, path="schematic_row.pdf"):
    fig.savefig(path, bbox_inches="tight")
```

---

## Example Script (`examples/build_full_figure.py`)

```python
from figure import build_row, export

if __name__ == "__main__":
    fig, axes = build_row(order=(1,3,2,4,5,6), annotate_axis=True)
    export(fig, "ca1_models_row.pdf")
    export(fig, "ca1_models_row.svg")
    print("Saved ca1_models_row.*")
```

---

## Notes on Concept Axis

Axis title: **“Spatially constrained → Unconstrained”** (preferred).  
Optional sublabels: **External → Internal → Peer (non-spatial)**.  
Recommended ordering: **1 → 3 → 2 → 4 → 5 → 6**.

---

## License

MIT — feel free to reuse and adapt.
