# How it works, in more detail

This expands on the README. It is still meant to be readable without opening the code.

## The seven stages

| Stage | What it does |
| --- | --- |
| `scaffold` | Reads the already-registered ST coordinates and GC identities. Writes nothing back. |
| `if` | Reads each CZI onto the ST grid, removes the stitching ripple, cuts the tissue outline, segments GC objects. |
| `align` | Brute-force orientation search, then local refinement, then ICP on GC centroids. |
| `match` | Pairs GCs, fits the elastic warp, gives every GC one identity across all sections. |
| `columns` | Stacks each GC's outlines into a smooth closed 3D surface. |
| `viewer` | Writes the standalone HTML. |
| `validate` | Collects QC numbers, hashes the outputs. |

## Everything happens on one grid

The first thing the pipeline does with a CZI is resample it to the ST bin size — 10 µm by
default. Both modalities are on the same ruler from that moment on, so no stage ever has
to reason about two different pixel sizes.

This also means the scale factor between the two modalities is **not** a free parameter:
it comes from the microscope metadata and the chip pitch, both of which are known exactly.
The orientation search only has to solve for rotation, mirror and translation, with a
small ±6% scale refinement for section stretch.

## Segmenting germinal centres on the IF side

The GC marker channel is thresholded with hysteresis: a strict threshold (Otsu) finds
confident GC cores, a looser one grows them outwards, and only the grown regions that
contain a confident core are kept. This picks up dimmer GCs without letting background
blobs in.

Before that, two corrections:

- **Stitching ripple.** Tiled acquisition leaves a periodic brightness ripple (~774 µm in
  our data, 13–20% of background). It is divided out along both axes using the median
  brightness of tissue pixels.
- **Tissue outline.** Computed from a weighted blend of channels, smoothed enough to see
  across the ripple, then closed, hole-filled, and reduced to the single largest component
  — a section is one piece of tissue.

A note from our data worth repeating: **DAPI was not usable as a common channel.** Its
brightness varied more than 13-fold between sections and saturated on three of five, so
nuclear segmentation was not attempted. The marker channels were stable across all five.
If your DAPI is well-behaved you have an extra channel to work with; do not assume it.

## Why the orientation search is scored on GCs

Scoring candidates by tissue overlap picks the wrong mirror when the section is roughly
convex. Concretely, for one of our sections:

| Candidate | Tissue Dice | GC centres within 150 µm |
| --- | --- | --- |
| correct (no mirror, 354°) | 0.927 | **40** |
| wrong (mirror, 315°) | 0.928 | 13 |

Tissue shape ranked the wrong answer first. The GC criterion is not close.

The pipeline therefore keeps several diverse candidates from the coarse search, carries
each through refinement and ICP, and only then picks the winner — on GC agreement.

## Choosing the stiffness of the warp

A thin-plate spline has one knob: how much it is allowed to bend. The pipeline sweeps it
across several orders of magnitude and, for each setting, runs leave-one-out
cross-validation over the GC anchors — hide one, fit on the rest, measure how far the
prediction lands from the hidden one. The setting with the lowest held-out error wins.

This is what stops the warp from overfitting. A very floppy spline drives the *fitted*
residual to zero while getting held-out GCs badly wrong; cross-validation sees through
that. In our data it selected stiff settings (10⁵–10⁶) with held-out errors of 14–31 µm.

## Turning nine outlines into a surface

Naively stacking nine masks gives a ziggurat. Two things fix it:

1. **Blur the volume, not the mesh.** The stack is oversampled in z, blurred with a
   Gaussian, and the surface is drawn through the blurred volume. The iso-surface of a
   blurred stack is smooth by construction.
2. **Solve for the level.** Rather than drawing the surface at a fixed halfway value, the
   pipeline finds the level at which the enclosed volume equals the volume actually
   measured from the sections. Smoothing then changes the shape's *appearance* without
   changing its *size*.

Two traps we hit, recorded so you do not:

- **Do not run a mesh smoother** (Taubin, Laplacian) on columns this thin. It collapsed
  ours to 1–15% of their true volume.
- **Pad the volume by at least 4σ on every axis** before drawing the surface. Otherwise
  the surface is clipped at the array border and the mesh is not closed — we had ~100 open
  edges per mesh until we did.

## What the QC report contains

`validation_report.json` records, per section: the chosen orientation and whether all
sections agreed on handedness; how many GCs matched and how closely; the warp's
cross-validation error; and the two-sided disagreement. Per GC: volume, maximum radius,
radius variability, centroid drift across the stack. Plus the mesh checks — watertightness
and mesh-versus-measured volume — and the standing caveats.

## Check the IF coverage before reading anything into the 3D shapes

`validation_report.json` reports, for each IF section, what fraction of the spatial
transcriptomics tissue it actually reaches, and how much of the shortfall lies outside the
imaged field altogether.

This matters more than it sounds. A germinal centre can only be matched where the IF
section covers it, so a section that is torn, folded, or scanned over a smaller field than
the chip will silently produce GCs that look like they "disappear" on that plane. In our
two replicates, only 68% and 75% of the transcriptomics tissue was covered by all five IF
sections, and about a quarter of the GCs that were present on every transcriptomics
section were missing from at least one IF plane as a result.

Lowering `tissue_threshold_scale` recovers dim tissue, but it cannot recover tissue that
is not there. A quick way to tell the two apart: sweep the threshold down and watch the
number of detected GC objects. If coverage rises while the GC count stays flat, the extra
area is empty and the missing GCs are a sectioning or imaging limitation, not a
thresholding one. That is exactly what we saw — coverage 75% to 83%, GC count 278 to 277.
