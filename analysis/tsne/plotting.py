"""Static Matplotlib and dependency-free interactive HTML t-SNE plots."""

from __future__ import annotations

import json
import math
import matplotlib.pyplot as plt

from dataclasses import asdict
from matplotlib.colors import to_hex, to_rgba
from pathlib import Path
from typing import Sequence

from .core import compute_tsne
from .tsne_types import TSNEConfig, TSNEGroup, TSNEResult


def save_static_figure(
    result: TSNEResult,
    output_path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 7.0),
    dpi: int = 300,
    show_axes: bool = False,
    show_legend: bool = True,
) -> Path:
    """Save a publication-friendly static figure and return its path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=figsize)
    for group in result.groups:
        axes.scatter(
            group.coordinates[:, 0],
            group.coordinates[:, 1],
            label=f"{group.name} (n={len(group.coordinates)})",
            c=group.color,
            marker=group.marker,
            s=group.marker_size,
            alpha=group.alpha,
            edgecolors=group.edge_color,
            linewidths=group.edge_width,
        )

    if title:
        axes.set_title(title)
    if show_axes:
        axes.set_xlabel("t-SNE 1")
        axes.set_ylabel("t-SNE 2")
    else:
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)
    if show_legend:
        axes.legend(frameon=False, loc="best")
    axes.margins(0.08)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _css_color(color: str, alpha: float = 1.0) -> str:
    red, green, blue, base_alpha = to_rgba(color)
    return (
        f"rgba({round(red * 255)}, {round(green * 255)}, {round(blue * 255)}, "
        f"{base_alpha * alpha:.4f})"
    )


def _html_json(value: object) -> str:
    # Escaping these characters prevents user-provided hover strings from
    # terminating the script tag while keeping Unicode paths readable.
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def save_interactive_html(
    result: TSNEResult,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save a small, standalone interactive plot without a Plotly dependency.

    The HTML contains only the viewer code, projected 2-D coordinates, styles,
    and hover strings. Original high-dimensional features are not embedded.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": title or "Interactive t-SNE",
        "groups": [
            {
                "name": group.name,
                "color": _css_color(group.color, group.alpha),
                "solidColor": to_hex(group.color),
                "edgeColor": _css_color(group.edge_color, group.alpha)
                if group.edge_color.lower() != "none"
                else "transparent",
                "edgeWidth": group.edge_width,
                "marker": group.marker,
                "radius": max(2.0, math.sqrt(group.marker_size) * 0.58),
                "points": [
                    [round(float(x), 7), round(float(y), 7), hover]
                    for (x, y), hover in zip(group.coordinates, group.hover_texts)
                ],
            }
            for group in result.groups
        ],
        "meta": {
            "selectedSamples": result.total_selected_samples,
            "inputSamples": result.total_input_samples,
            "featureDimension": result.feature_dimension,
            "config": asdict(result.config),
        },
    }

    html = _INTERACTIVE_HTML.replace("__TSNE_PAYLOAD__", _html_json(payload))
    output_path.write_text(html, encoding="utf-8")
    return output_path


def plot_tsne_groups(
    groups: Sequence[TSNEGroup],
    output_path: str | Path,
    *,
    config: TSNEConfig | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 7.0),
    dpi: int = 300,
    show_axes: bool = False,
    show_legend: bool = True,
) -> TSNEResult:
    """Compute one shared projection, always save a static plot, and optionally HTML."""
    print('Computing T-SNE...')
    output_path = Path(output_path)
    result = compute_tsne(groups, config)
    save_static_figure(
        result,
        output_path,
        title=title,
        figsize=figsize,
        dpi=dpi,
        show_axes=show_axes,
        show_legend=show_legend,
    )

    save_interactive_html(result, output_path.with_suffix(".html"), title=title)
    return result


_INTERACTIVE_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Interactive t-SNE</title>
<style>
  :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
  * { box-sizing: border-box; }
  html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #f7f8fa; }
  #app { width: 100%; height: 100%; display: grid; grid-template-rows: auto 1fr; }
  header { display: flex; align-items: center; gap: 16px; padding: 10px 16px; background: #fff;
    border-bottom: 1px solid #dfe3e8; box-shadow: 0 1px 5px rgba(0,0,0,.04); z-index: 2; }
  h1 { margin: 0; font-size: 17px; font-weight: 650; white-space: nowrap; }
  #hint { color: #667085; font-size: 12px; flex: 1; }
  button { appearance: none; border: 1px solid #cfd5dd; border-radius: 6px; background: #fff;
    padding: 6px 10px; cursor: pointer; color: #344054; }
  button:hover { background: #f2f4f7; }
  main { position: relative; min-height: 0; }
  canvas { display: block; width: 100%; height: 100%; background: #fff; cursor: grab; outline: none; }
  canvas.dragging { cursor: grabbing; }
  #legend { position: absolute; right: 14px; top: 14px; min-width: 150px; max-width: 300px;
    padding: 10px 12px; border: 1px solid #e0e4e9; border-radius: 8px; background: rgba(255,255,255,.92);
    box-shadow: 0 3px 12px rgba(16,24,40,.08); font-size: 12px; backdrop-filter: blur(6px); }
  .legend-row { display: grid; grid-template-columns: 12px 1fr auto; align-items: center; gap: 7px;
    padding: 4px 2px; cursor: pointer; user-select: none; }
  .legend-row.off { opacity: .35; text-decoration: line-through; }
  .swatch { width: 10px; height: 10px; border-radius: 50%; }
  .count { color: #667085; }
  #tooltip { position: fixed; display: none; z-index: 5; max-width: min(560px, 70vw); max-height: 45vh;
    overflow: auto; pointer-events: none; padding: 8px 10px; border-radius: 6px; color: #fff;
    background: rgba(17,24,39,.94); box-shadow: 0 5px 18px rgba(0,0,0,.18); white-space: pre-wrap;
    overflow-wrap: anywhere; font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1 id="title"></h1>
    <div id="hint">Wheel / + −: zoom · drag / arrow keys: pan · hover: inspect · R or double-click: reset</div>
    <button id="reset" type="button">Reset view</button>
  </header>
  <main>
    <canvas id="plot" tabindex="0" aria-label="Interactive t-SNE plot"></canvas>
    <div id="legend"></div>
    <div id="tooltip"></div>
  </main>
</div>
<script>
"use strict";
const data = __TSNE_PAYLOAD__;
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const legend = document.getElementById("legend");
document.getElementById("title").textContent = data.title;
document.title = data.title;
data.groups.forEach(group => group.visible = true);

const allPoints = data.groups.flatMap((group, groupIndex) =>
  group.points.map((point, pointIndex) => ({ group, groupIndex, pointIndex, x: point[0], y: point[1], text: point[2] }))
);
let bounds, view, dragging = false, dragStart = null, hoverPoint = null;

function calculateBounds() {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const point of allPoints) {
    minX = Math.min(minX, point.x); maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y); maxY = Math.max(maxY, point.y);
  }
  const spanX = Math.max(maxX - minX, 1), spanY = Math.max(maxY - minY, 1);
  return { minX: minX - spanX * .08, maxX: maxX + spanX * .08,
           minY: minY - spanY * .08, maxY: maxY + spanY * .08 };
}
function resetView() { bounds = calculateBounds(); view = {...bounds}; tooltip.style.display = "none"; draw(); }
function resize() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}
function dimensions() { const r = canvas.getBoundingClientRect(); return {w:r.width, h:r.height}; }
function toScreen(x, y) {
  const {w, h} = dimensions();
  return [(x - view.minX) / (view.maxX - view.minX) * w,
          h - (y - view.minY) / (view.maxY - view.minY) * h];
}
function toWorld(x, y) {
  const {w, h} = dimensions();
  return [view.minX + x / w * (view.maxX - view.minX),
          view.minY + (h - y) / h * (view.maxY - view.minY)];
}
function markerPath(marker, x, y, radius) {
  ctx.beginPath();
  if (["s", "square"].includes(marker)) ctx.rect(x-radius, y-radius, radius*2, radius*2);
  else if (["^", "triangle-up"].includes(marker)) { ctx.moveTo(x,y-radius*1.25); ctx.lineTo(x+radius*1.1,y+radius); ctx.lineTo(x-radius*1.1,y+radius); ctx.closePath(); }
  else if (["D", "d", "diamond"].includes(marker)) { ctx.moveTo(x,y-radius*1.25); ctx.lineTo(x+radius*1.05,y); ctx.lineTo(x,y+radius*1.25); ctx.lineTo(x-radius*1.05,y); ctx.closePath(); }
  else if (["x", "+"].includes(marker)) { ctx.moveTo(x-radius,y-radius); ctx.lineTo(x+radius,y+radius); ctx.moveTo(x+radius,y-radius); ctx.lineTo(x-radius,y+radius); }
  else ctx.arc(x, y, radius, 0, Math.PI*2);
}
function draw() {
  if (!view || !canvas.width) return;
  const {w, h} = dimensions();
  ctx.clearRect(0, 0, w, h);
  for (const group of data.groups) {
    if (!group.visible) continue;
    ctx.fillStyle = group.color; ctx.strokeStyle = group.edgeColor; ctx.lineWidth = group.edgeWidth;
    for (const point of group.points) {
      const [x,y] = toScreen(point[0], point[1]);
      if (x < -20 || y < -20 || x > w+20 || y > h+20) continue;
      markerPath(group.marker, x, y, group.radius);
      if (["x", "+"].includes(group.marker)) { ctx.strokeStyle = group.solidColor; ctx.stroke(); }
      else { ctx.fill(); if (group.edgeColor !== "transparent" && group.edgeWidth > 0) ctx.stroke(); }
    }
  }
  if (hoverPoint && hoverPoint.group.visible) {
    const [x,y] = toScreen(hoverPoint.x, hoverPoint.y);
    ctx.beginPath(); ctx.arc(x,y,hoverPoint.group.radius+4,0,Math.PI*2);
    ctx.strokeStyle="#111827"; ctx.lineWidth=1.5; ctx.stroke();
  }
}
function zoomAt(factor, screenX, screenY) {
  const [wx, wy] = toWorld(screenX, screenY);
  view.minX = wx + (view.minX-wx)*factor; view.maxX = wx + (view.maxX-wx)*factor;
  view.minY = wy + (view.minY-wy)*factor; view.maxY = wy + (view.maxY-wy)*factor;
  draw();
}
function panPixels(dx, dy) {
  const {w,h} = dimensions(), sx=(view.maxX-view.minX)/w, sy=(view.maxY-view.minY)/h;
  view.minX -= dx*sx; view.maxX -= dx*sx; view.minY += dy*sy; view.maxY += dy*sy; draw();
}
function nearestPoint(x, y) {
  let best=null, bestDistance=Infinity;
  for (const point of allPoints) {
    if (!point.group.visible) continue;
    const [px,py]=toScreen(point.x,point.y), distance=Math.hypot(px-x,py-y);
    if (distance < bestDistance) { best=point; bestDistance=distance; }
  }
  return bestDistance <= Math.max(10, best ? best.group.radius+5 : 10) ? best : null;
}
function showHover(event) {
  const rect=canvas.getBoundingClientRect(), x=event.clientX-rect.left, y=event.clientY-rect.top;
  hoverPoint=nearestPoint(x,y); draw();
  if (!hoverPoint) { tooltip.style.display="none"; return; }
  tooltip.textContent = `${hoverPoint.group.name}\n${hoverPoint.text}`;
  tooltip.style.display="block";
  const margin=12, tw=tooltip.offsetWidth, th=tooltip.offsetHeight;
  tooltip.style.left=Math.min(event.clientX+14, window.innerWidth-tw-margin)+"px";
  tooltip.style.top=Math.min(event.clientY+14, window.innerHeight-th-margin)+"px";
}

canvas.addEventListener("wheel", event => { event.preventDefault(); const r=canvas.getBoundingClientRect(); zoomAt(event.deltaY < 0 ? .82 : 1.22,event.clientX-r.left,event.clientY-r.top); }, {passive:false});
canvas.addEventListener("pointerdown", event => { dragging=true; dragStart=[event.clientX,event.clientY]; canvas.classList.add("dragging"); canvas.setPointerCapture(event.pointerId); canvas.focus(); });
canvas.addEventListener("pointermove", event => { if (dragging) { panPixels(event.clientX-dragStart[0],event.clientY-dragStart[1]); dragStart=[event.clientX,event.clientY]; tooltip.style.display="none"; } else showHover(event); });
canvas.addEventListener("pointerup", event => { dragging=false; canvas.classList.remove("dragging"); canvas.releasePointerCapture(event.pointerId); });
canvas.addEventListener("pointerleave", () => { if (!dragging) { hoverPoint=null; tooltip.style.display="none"; draw(); } });
canvas.addEventListener("dblclick", resetView);
canvas.addEventListener("keydown", event => {
  const key=event.key.toLowerCase(), {w,h}=dimensions();
  if (["+","="].includes(key)) zoomAt(.82,w/2,h/2);
  else if (["-","_"].includes(key)) zoomAt(1.22,w/2,h/2);
  else if (key==="arrowleft") panPixels(35,0); else if (key==="arrowright") panPixels(-35,0);
  else if (key==="arrowup") panPixels(0,35); else if (key==="arrowdown") panPixels(0,-35);
  else if (key==="r") resetView(); else return;
  event.preventDefault();
});
document.getElementById("reset").addEventListener("click", resetView);
for (const group of data.groups) {
  const row=document.createElement("div"); row.className="legend-row";
  const swatch=document.createElement("span"); swatch.className="swatch"; swatch.style.background=group.solidColor;
  const label=document.createElement("span"); label.textContent=group.name;
  const count=document.createElement("span"); count.className="count"; count.textContent=`n=${group.points.length}`;
  row.append(swatch,label,count); row.addEventListener("click",()=>{group.visible=!group.visible;row.classList.toggle("off",!group.visible);hoverPoint=null;tooltip.style.display="none";draw();});
  legend.appendChild(row);
}
new ResizeObserver(resize).observe(canvas);
resetView(); resize(); canvas.focus();
</script>
</body>
</html>
'''
