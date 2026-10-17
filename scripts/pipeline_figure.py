"""Generate the manuscript pipeline as standalone HTML/SVG; dimensions are in PDF points."""
from pathlib import Path

W = 816
G = 16                       # arrow gap
nodes, wires = [], []

CSS = """
.pipe { position: relative; width: 816pt; color: #334155; line-height: 1.3; }
.pipe .wire { position: absolute; left: 0; top: 0; width: 100%; height: 100%; overflow: visible; }
.pipe .wire text { font-size: 14px; font-weight: 700; }
.pipe .n { position: absolute; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 2pt;
  text-align: center; font-size: 16pt; border: 2pt solid; border-radius: 10pt; padding: 0 14pt; }
.pipe .n.bare { border: 0; padding: 0; }
.pipe .n b { font-size: 20pt; color: #0f172a; }
.pipe .n.mini { font-size: 15pt; padding: 0 8pt; }
.pipe .n.mini b { font-size: 17pt; }
.pipe .n.dia b { font-size: 18pt; }
.pipe .n.dia { font-size: 15pt; }
.pipe .num { position: absolute; left: 10pt; top: 10pt; width: 26pt; height: 26pt; border-radius: 50%; color: #fff;
  font-size: 15pt; font-weight: 700; line-height: 26pt; text-align: center; }
.pipe .frame { position: absolute; border: 2pt dashed #065f46; border-radius: 14pt; background: #fff; }
.pipe .tag { position: absolute; height: 24pt; line-height: 24pt; padding: 0 14pt; border-radius: 12pt; background: #065f46; color: #fff;
  font-size: 13.5pt; font-weight: 700; letter-spacing: .04em; }
.pipe mjx-container { margin: 0 !important; }
"""


def node(x, y, w, h, title, lines=(), fill="#fff", stroke="#334155", num=None, cls=""):
    body = (f"<b>{title}</b>" if title else "") + "".join(f"<div>{ln}</div>" for ln in lines)
    badge = f'<span class="num" style="background:{stroke}">{num}</span>' if num else ""
    nodes.append(f'<div class="n {cls}" style="left:{x}pt;top:{y}pt;width:{w}pt;height:{h}pt;'
                 f'background:{fill};border-color:{stroke}">{badge}{body}</div>')
    return y + h


def box(x, y, w, title, lines=(), fill="#fff", stroke="#334155", num=None, h=None):
    return node(x, y, w, h or (64 if len(lines) <= 1 else 64 + 23 * (len(lines) - 1)), title, lines, fill, stroke, num)


def diamond(cx, cy, hw, hh, title, lines=()):
    wires.append(f'<polygon points="{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}" fill="#fefce8" stroke="#a16207" stroke-width="2"/>')
    node(cx - hw * .6, cy - hh, hw * 1.2, 2 * hh, title, lines, "transparent", "transparent", cls="bare dia")


MK = {"#475569": "pa", "#c2410c": "pe", "#065f46": "pw"}


def arrow(pts, color="#475569", dash=False, label=None, lx=0, ly=0, rot=False):
    p = " ".join(f"{x},{y}" for x, y in pts)
    d = ' stroke-dasharray="9 6"' if dash else ""
    wires.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2.5"{d} stroke-linejoin="round" marker-end="url(#{MK[color]})"/>')
    if label:
        t = f' transform="rotate(-90 {lx} {ly})"' if rot else ""
        wires.append(f'<text x="{lx}" y="{ly}" fill="{color}" text-anchor="middle"{t}>{label}</text>')


def down(x, y0, y1):
    arrow([(x, y0), (x, y1 - 2)])
    return y1


L, R, MID, LC = 40, 776, 408, 230

# 1 problem setup
y = box(4, 4, 808, "Problem setup", [
    r"$\dot x=-0.1\,yz,\quad \dot y=1.6\,xz,\quad \dot z=-0.75\,xy,\qquad u_0=(0.5,\,0.75,\,1.0)$",
    r"$t\in[0,\,13.26446]$ · 27 windows · 1,536 points/window · float64"], "#fff7ed", "#c2410c", 1)
frame_top = y + 16
y = down(MID, y, frame_top + 22)

# 2 PINN for window k
y2 = y
y = node(L, y, R - L, 132, "", (), "#eff6ff", "#1e40af", 2)
nodes.append(f'<div class="n bare" style="left:{L}pt;top:{y2 + 12}pt;width:{R - L}pt;height:30pt"><b>PINN for window $k$</b></div>')
x = 58
for i, (w, t, d) in enumerate([(92, "Input", "$t$"), (132, "Hidden", "4×60 tanh"),
                               (140, "Output", r"$N_k(t)\in\mathbb{R}^3$"), (264, "Hard IC", r"$\hat u_k=u_{0,k}+(t-t_k)\,N_k(t)$")]):
    node(x, y2 + 50, w, 70, t, [d], "#fff", "#93c5fd", cls="mini")
    if i < 3:
        arrow([(x + w + 3, y2 + 85), (x + w + 23, y2 + 85)])
    x += w + 26
mid2 = y2 + 66

# 3-5 residual, causal loss, Adam
y = down(MID, y, y + G)
y = box(L, y, R - L, "ODE residual", [r"$r_k=d\hat u_k/dt-f(\hat u_k)$, derivatives by autodiff"], "#f5f3ff", "#6d28d9", 3)
y = down(MID, y, y + G)
y4 = y
y = box(L, y, R - L, "Causal residual loss", [
    r"$\mathcal L_i=\mathrm{mean}(r_i^2),\quad w_i=\exp\left(-\varepsilon\textstyle\sum_{j\lt i}\mathcal L_j\right),\quad \mathcal L=\mathrm{mean}(w_i\mathcal L_i)$",
    r"$\varepsilon:\ 0.01\to 0.1\to 1\to 10$"], "#f5f3ff", "#6d28d9", 4)
mid4 = (y4 + y) / 2
y = down(MID, y, y + G)
y = box(L, y, R - L, "Adam optimization", ["learning rate × 0.9 every 1,000 steps"], "#f0fdfa", "#0d9488", 5)
arrow([(LC, y), (LC, y + G - 2)])

# stage / epsilon decisions
hh = 56
cy = y + G + hh
diamond(LC, cy, 190, hh, "Stage complete?", [r"$\min w>0.99$", "or 4,000 steps"])
diamond(615, cy, 145, hh, r"All $\varepsilon$ values used?")
arrow([(420, cy), (468, cy)], label="YES", lx=444, ly=cy - 9)
arrow([(40, cy), (22, cy), (22, mid4), (38, mid4)], label="NO", lx=40, ly=cy + 26)
row = cy + hh + 26
arrow([(615, cy + hh), (615, row - 2)], label="NO", lx=637, ly=row - 6)
arrow([(542, cy + hh / 2), (542, row - 12), (LC, row - 12), (LC, row - 2)], label="YES", lx=390, ly=row - 18)
box(480, row, 270, r"Next $\varepsilon$", ["keep weights"], "#fff7ed", "#c2410c")
arrow([(750, row + 32), (788, row + 32), (788, mid4), (778, mid4)], "#c2410c", True, "next ε", 780, (mid4 + row) / 2, True)

# 6-7 polish and save
y = box(L, row, 380, "L-BFGS", ["≤ 1,000 iterations", "unweighted residual MSE"], "#f0fdfa", "#0d9488", 6)
y = down(LC, y, y + G)
y = box(L, y, 380, "Save window", ["checkpoint · history · prediction"], "#f8fafc", "#475569", 7)
arrow([(LC, y), (LC, y + G - 2)])

# window decision and handoff
hh = 50
cy = y + G + hh
diamond(LC, cy, 180, hh, "Last window?", ["$k=27$"])
box(470, cy - 44, 306, "Next window", [r"$u_{0,k+1}=\hat u_k(t_{k+1})$", r"copy weights, $k\leftarrow k+1$"], "#ecfdf5", "#065f46", h=88)
arrow([(410, cy), (468, cy)], label="NO", lx=439, ly=cy - 9)
arrow([(776, cy), (800, cy), (800, mid2), (778, mid2)], "#065f46", True, "next window", 794, (mid2 + mid4) / 2, True)
frame_bot = cy + hh + 18
y = frame_bot + 20
arrow([(LC, cy + hh), (LC, y - 2)], label="YES", lx=258, ly=frame_bot - 4)

# 8-9 evaluation
y = box(4, y, 808, "Reference evaluation", [
    r"DOP853 · 13,265 points · rtol $10^{-10}$ · atol $10^{-12}$",
    "used only for evaluation, never in training"], "#eff6ff", "#1e40af", 8)
y = down(MID, y, y + G)
y = box(4, y, 808, "Evaluation outputs",
        ["MAE · RMSE · max error · residual MSE · invariant drift · training cost"], "#f8fafc", "#475569", 9)
H = y + 4

markers = "".join(
    f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c, i in MK.items())
frame = (f'<div class="frame" style="left:4pt;top:{frame_top}pt;width:808pt;height:{frame_bot - frame_top}pt"></div>'
         f'<div class="tag" style="left:24pt;top:{frame_top - 12}pt">REPEAT FOR EACH WINDOW <i>k</i> = 1 … 27</div>')
pipe = (f'<style>{CSS}</style><div class="pipe" style="height:{H}pt">{frame}'
        f'<svg class="wire" viewBox="0 0 {W} {H}"><defs>{markers}</defs>{"".join(wires)}</svg>{"".join(nodes)}</div>')

final = Path(__file__).resolve().parents[1] / "figures"
final.mkdir(exist_ok=True)
(final / "pipeline.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Training pipeline</title>
<script>window.MathJax={{tex:{{inlineMath:[["$","$"]]}},svg:{{fontCache:"global"}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>@page {{ size: {W}pt {H}pt; margin: 0; }} * {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: "Segoe UI", sans-serif; background: #fff; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}</style>
</head><body>{pipe}</body></html>""", encoding="utf-8")
