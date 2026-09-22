#!/usr/bin/env python3
"""Four-slice RCTD cell-type atlas — same visual language as the v4 GC-matching HTML.

Two selection rules, deliberately:
  * DZ, LZ, FDC, GC-Tfh  -> normalised RCTD weight >= 0.15. This reproduces the
    existing DZ_LZ_clean plots spot-for-spot.
  * B follicle           -> first_type in {B naive, B memory, B MZ}. Summing the
    weights of three types does NOT work here: RCTD ran in doublet mode, so many
    red-pulp spots carry a modest B-cell second_type and the sum crosses any
    threshold while the spot is really erythrocyte. Requiring the B cell to be the
    dominant call is what produces a follicle instead of a haze.

    B follicle  pale blue | DZ green | LZ white | FDC purple | GC-Tfh gold
GC context comes from the v4 curated regions, so hover reports the same GC ids
used by the four-slice atlas and the nine-slice viewers.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import h5py, numpy as np, pandas as pd, plotly.graph_objects as go

THRESH = 0.15
Z_BY_SLICE = [0, 250, 500, 750]          # Z_SPACING 50 * Z_SCALE 5, as in the v4 atlas
GC_CONTEXT_STRIDE = 3                     # thin the GC backdrop; hover still lands on a GC

FOLLICLE_TYPES = ("B naive", "B memory", "B MZ")
DOT = 1                      # same marker size as the GC spots in the v4 atlas
# Draw order matters: compartments first, GC internals painted on top.
# (key, label, colour, opacity, rule, gc_only)
LAYERS = [
    ("B_foll",  "B follicle  (naive + memory + MZ)", "#9ECBFF", 0.55, "first_type", False),
    ("B_pre",   "pre-B",                             "#FF8FD0", 0.80, "first_type", False),
    ("CD4_T",   "CD4 T  (naive)",                    "#FF7A3D", 0.70, "first_type", False),
    ("CD8_T",   "CD8 T  (effector)",                 "#18D6D6", 0.80, "first_type", False),
    ("TRC",     "TRC",                               "#E0C097", 0.80, "first_type", False),
    ("DZ",      "DZ",                                "#2ECC40", 0.95, "weight",     True),
    ("LZ",      "LZ",                                "#FFFFFF", 0.95, "weight",     True),
    ("FDC",     "FDC",                               "#A855F7", 1.00, "weight",     True),
    ("GC_Tfh",  "GC-Tfh",                            "#FFC814", 1.00, "weight",     True),
]
FIRST_TYPE_SETS = {
    "B_foll": FOLLICLE_TYPES,
    "B_pre":  ("B pre",),
    "CD4_T":  ("CD4+ Naive",),
    "CD8_T":  ("CD8+ T effector",),
    "TRC":    ("TRC",),
}


def selected(d, key, rule, gc_only):
    """Boolean mask for one display layer.

    The GC zone and niche types are shown only where the curated v4 GC region
    says there is a germinal centre; free-floating calls elsewhere in the
    section are deconvolution noise for this purpose and are dropped.
    """
    if rule == "first_type":
        m = d.first_type.isin(FIRST_TYPE_SETS[key])
    else:
        m = d[key] >= THRESH
    return (m & (d.gc > 0)) if gc_only else m
REPS = {
    "rep1": dict(
        h5ad="/path/to/rep1_isotype/"
             "merged_gc/global_bin20_alignment/v4_size_ranked/four_slice_gc_matching_alignment_v4.h5ad",
        slices=["B06289D1", "B06289E3", "B06289G4", "B06289G5"],
        short=["D1", "E3", "G4", "G5"]),
    "rep2": dict(
        h5ad="/path/to/rep2_isotype/"
             "merged_gc/global_bin20_alignment/v4_size_ranked/four_slice_gc_matching_alignment_v4.h5ad",
        slices=["B06286A4", "B06286C1", "B06286C4", "B06286C5"],
        short=["A4", "C1", "C4", "C5"]),
}


def cats(handle, name):
    node = handle["obs"][name]
    if isinstance(node, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in node["categories"][:]])
        k = node["codes"][:]
        return np.where(k >= 0, c[np.clip(k, 0, None)], "NA")
    return node[:]


def load_rep(rep, weights_dir):
    cfg = REPS[rep]
    with h5py.File(cfg["h5ad"], "r") as h:
        obs = pd.DataFrame({
            "slice": cats(h, "slice_name"),
            "gc": h["obs"]["gc_match_curated_final_id"][:].astype(np.int32),
        })
        obs["sx"] = h["obsm"]["spatial"][:, 0].astype(np.int32)
        obs["sy"] = h["obsm"]["spatial"][:, 1].astype(np.int32)
        ali = h["obsm"]["align_deformable"][:]
        obs["ax"] = ali[:, 0].astype(np.float32)
        obs["ay"] = ali[:, 1].astype(np.float32)
    frames = []
    for i, s in enumerate(cfg["slices"]):
        w = pd.read_csv(Path(weights_dir) / f"{rep}_{s}_rctd8.csv.gz")
        w = w.rename(columns={"x": "sx", "y": "sy"})
        part = obs[obs["slice"] == s]
        merged = part.merge(w, on=["sx", "sy"], how="inner")
        merged["z"] = Z_BY_SLICE[i]
        merged["short"] = cfg["short"][i]
        print(f"  {s}: v4 {len(part):,} | rctd {len(w):,} | joined {len(merged):,}"
              f" ({100*len(merged)/len(w):.1f}% of rctd)")
        if len(merged) < 0.9 * min(len(part), len(w)):
            raise SystemExit(f"join too small for {s} — coordinate systems disagree")
        frames.append(merged)
    return pd.concat(frames, ignore_index=True), cfg


def build(rep, weights_dir, out_html):
    print(f"[{rep}] loading")
    df, cfg = load_rep(rep, weights_dir)
    traces, groups = [], []

    # --- GC context: every curated GC bin, dim, carries the GC id on hover ---
    gc = df[df.gc > 0]
    for i, sh in enumerate(cfg["short"]):
        d = gc[gc.short == sh].iloc[::GC_CONTEXT_STRIDE]
        traces.append(go.Scatter3d(
            x=d.ax, y=d.ay, z=d.z, mode="markers", name="GC region (v4 curated)",
            legendgroup="gc", showlegend=(i == 0),
            marker=dict(size=DOT, color="#5A6478", opacity=0.32),
            customdata=np.stack([d.gc.values, np.full(len(d), sh)], axis=-1),
            hovertemplate="GC%{customdata[0]} · %{customdata[1]}<extra></extra>"))
    groups.append(("gc", "GC region"))

    # --- one trace per cell type per slice ---
    for key, label, colour, op, rule, gc_only in LAYERS:
        for i, sh in enumerate(cfg["short"]):
            part = df[df.short == sh]
            d = part[selected(part, key, rule, gc_only)]
            traces.append(go.Scatter3d(
                x=d.ax, y=d.ay, z=d.z, mode="markers", name=label,
                legendgroup=key, showlegend=(i == 0),
                marker=dict(size=DOT, color=colour, opacity=op),
                customdata=np.stack([
                    (d[key].values if rule == "weight" else d.first_type.values),
                    d.gc.values, np.full(len(d), sh)], axis=-1),
                hovertemplate=(
                    f"{label}<br>weight %{{customdata[0]:.2f}}"
                    "<br>%{customdata[2]} · GC%{customdata[1]}<extra></extra>"
                    if rule == "weight" else
                    f"{label}<br>dominant call %{{customdata[0]}}"
                    "<br>%{customdata[2]} · GC%{customdata[1]}<extra></extra>")))
        groups.append((key, label))
        n = int(selected(df, key, rule, gc_only).sum())
        print(f"  {label}: {n:,} spots ({rule}{', inside GC only' if gc_only else ''})")

    n_slices = len(cfg["short"])
    n_groups = len(groups)

    def vis(active):
        out = []
        for g, _ in groups:
            out.extend([g in active] * n_slices)
        return out

    allk = [g for g, _ in groups]
    buttons = [dict(label="All layers", method="update",
                    args=[{"visible": vis(set(allk))},
                          {"title.text": f"{rep} · RCTD cell types — all layers"}])]
    for g, lab in groups:
        # every single-type view keeps the GC region as a dim backdrop
        shown = {g} if g == "gc" else {"gc", g}
        buttons.append(dict(label=lab, method="update",
                            args=[{"visible": vis(shown)}, {"title.text": f"{rep} · {lab}"}]))

    pad = 400
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"{rep} · RCTD cell types across four aligned sections",
                   font=dict(size=20, color="white"), x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(title=dict(text=""), showgrid=True, gridcolor="#333333",
                       range=[df.ax.min() - pad, df.ax.max() + pad]),
            yaxis=dict(title=dict(text=""), showgrid=True, gridcolor="#333333",
                       range=[df.ay.min() - pad, df.ay.max() + pad]),
            zaxis=dict(title=dict(text=""), showgrid=True, gridcolor="#333333",
                       range=[-37.5, 787.5]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0)),
            aspectmode="data", bgcolor="black"),
        paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0.7)", bordercolor="white",
                    borderwidth=1, x=1.0, y=1.0, xanchor="right", yanchor="top",
                    itemsizing="constant"),
        updatemenus=[dict(type="buttons", bgcolor="rgba(239,246,255,0.98)",
                          bordercolor="#2563eb", borderwidth=2, active=0,
                          buttons=buttons, direction="down", showactive=True, pad=dict(t=2, b=2),
                          x=0.005, xanchor="left", y=0.98, yanchor="top",
                          font=dict(color="#0f172a", size=10.5))])

    # write with a dtype guard — Plotly rejects i8 and fails silently on the buttons
    html = fig.to_html(include_plotlyjs=True, full_html=True,
                       config=dict(displaylogo=False, responsive=True))
    bad = [d for d in ("i8", "u8") if f'"dtype":"{d}"' in html or f'"dtype": "{d}"' in html]
    if bad:
        raise SystemExit(f"refusing to write: unsupported plotly dtype {bad}")
    Path(out_html).write_text(html, encoding="utf-8")
    mb = Path(out_html).stat().st_size / 1e6
    print(f"  wrote {out_html} ({mb:.1f} MB, {len(traces)} traces, {len(buttons)} buttons)")
    return dict(rep=rep, traces=len(traces), buttons=len(buttons), mb=round(mb, 1),
                spots={k: int(selected(df, k, r, go).sum()) for k, _, _, _, r, go in LAYERS},
                rules={k: (r + ("+GC-only" if go else "")) for k, _, _, _, r, go in LAYERS},
                gc_bins=int((df.gc > 0).sum()), n_gc=int(df.loc[df.gc > 0, "gc"].nunique()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=list(REPS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = build(a.rep, a.weights, a.out)
    print(json.dumps(rec, indent=1))
