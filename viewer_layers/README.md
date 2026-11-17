# Viewer layers

This folder holds the scripts that build the optional layers of the gcalign 3D
viewer from a finished run. The viewer works without any of them: it always
draws the germinal-centre columns and the sections, and it adds a layer only
when that layer's file is in the run directory.

Abbreviations used below: IF = immunofluorescence, ST = spatial
transcriptomics, GC = germinal centre, DZ / LZ = the dark zone and the light
zone of a GC, RCTD = robust cell-type decomposition (the method that gives each
ST bin a weight per cell type).

## The layers

| Layer in the viewer | File in the run directory | Built by |
|---|---|---|
| B-cell follicles: faint points behind the GCs on every section | `05_viewer/follicle_points.npz` | `build_follicle_layer.py` |
| DZ / LZ points: the measured dark-zone and light-zone points inside each GC | `05_viewer/dzlz_points.npz` | `build_dzlz_layer.py` |
| DZ / LZ solids: one dark-zone and one light-zone solid per GC | `04_columns/zone_blocks.npz` | `build_zone_blocks.py` |
| IF images: each IF section as a true-colour image with the GC outlines and ids | `05_viewer/if_images.json` | `build_if_images.py` |

`finish_all.sh` runs the four builders and then rebuilds the viewer.

<p align="center"><img src="../docs/img/follicle.jpg" width="100%"></p>

The picture shows the view "ST" of replicate 1 with the switch "B-cell follicles" on.
The pink outlines are the GCs on the four ST sections. The pale blue points are the
B-cell follicles on the same four sections. The switch draws the follicles of the
sections that the current view shows; the colour is the same on IF and on ST sections.

<p align="center"><img src="../docs/img/zone_blocks.jpg" width="100%"></p>

The picture shows the view "DZ / LZ" of replicate 1 with "Zones drawn as: Solids".
Each GC is cut into one dark-zone solid (green) and one light-zone solid (white).

## What the builders need

- **The experiment config and its finished run.** Every builder takes
  `--config my_experiment.yaml`. The run is read from the config's
  `output_dir`; `--output-dir` points a builder at a run that was moved or
  copied. The sections, their kinds (IF or ST) and their names come from the
  config, so nothing in the scripts names a particular experiment.
- **One marker channel per IF layer**, named exactly as in the config's
  `if_images.channels`:
  - `--follicle-channel`: the channel that marks the B-cell follicle (for
    example IgD).
  - `--zone-channel`: the channel that is high in the GC light zone (for
    example CD4, which marks the T follicular helper cells gathered there).
- **RCTD weight tables for the ST layers**, one CSV or CSV.gz file per ST
  section, in one directory given by `--weights`. `extract_rctd.R` writes these
  tables from RCTD `SpatialExperiment` objects. Each table needs the columns `x`
  and `y` (the bin's position on the chip grid, in DNB units), `first_type`
  (the dominant cell type) and one weight column per cell type of interest
  (`DZ` and `LZ` by default; see `--dz-column` and `--lz-column`).
  A builder finds the table of an ST section by the section's `source` in the
  config, which for an ST section is its chip id. With the default
  `--weights-pattern "*{source}*.csv*"`, a section with source `B06289D1` uses
  the one file whose name contains `B06289D1`. If several files match, pass a
  narrower pattern such as `--weights-pattern "*{source}_rctd8.csv.gz"`.

## Running them

```bash
# each builder on its own
python viewer_layers/build_follicle_layer.py --config my_experiment.yaml \
    --weights /path/to/rctd_weights --follicle-channel IgD_AF647
python viewer_layers/build_dzlz_layer.py --config my_experiment.yaml \
    --weights /path/to/rctd_weights --zone-channel CD4_AF594
python viewer_layers/build_zone_blocks.py --config my_experiment.yaml \
    --weights /path/to/rctd_weights --zone-channel CD4_AF594
python viewer_layers/build_if_images.py --config my_experiment.yaml

# or all four, then the viewer
FOLLICLE_CHANNEL=IgD_AF647 ZONE_CHANNEL=CD4_AF594 \
WEIGHTS_PATTERN='*{source}_rctd8.csv.gz' PYTHON=/path/to/python \
    viewer_layers/finish_all.sh my_experiment.yaml /path/to/rctd_weights
```

Every builder prints what it found on each section and writes a report next to
its output (`*_report.json`; the solids also get `zone_block_metrics.csv`, one
row per GC). `--help` on any script lists its options and their defaults.

## How each layer is made

**B-cell follicles.** On an IF section, the follicle is every pixel where the
follicle marker is above its Otsu threshold inside the tissue, after a small
morphological closing and removal of patches under 25 grid pixels. On an ST
section, the follicle is every bin whose dominant RCTD type is a follicular
B-cell type (`--follicle-types`, default `B naive,B memory,B MZ`). The dominant
type is used rather than a sum of weights: a summed B-cell weight also lights
up red-pulp bins that carry a minor B-cell share. Each section keeps at most
6,000 points, drawn at random with a fixed seed.

**DZ / LZ points.** On an IF section, each GC is split at its own median of the
light-zone marker: pixels at or above the median are light zone, the rest are
dark zone. The split is unsmoothed. In a sweep over smoothing (sigma 0, 1.2 and
2.5 pixels) and split point (50/50 to 80/20), smoothing lowered the agreement
with the neighbouring ST sections by up to 15 percentage points, while the split
point changed it very little, so the simplest rule was kept. On an ST section, a
bin inside a GC is dark zone or light zone when its normalised RCTD weight for
that zone is at least 0.15. The two zones are thresholded independently, so the
point clouds can overlap and leave gaps.

**DZ / LZ solids.** Each GC is cut into two solids that fill it exactly. A
light-zone score is set on every pixel of the GC: the light-zone marker on IF
sections, the RCTD light-zone weight minus the dark-zone weight on ST sections.
The score is z-scored inside the GC on each section, so the two kinds of
section share one scale. All sections are stacked at their true spacing and
smoothed together in 3D, so a noisy section takes its polarity from its
neighbours. One threshold then splits the whole GC, placed so that the
light-zone share equals the GC's light-zone fraction measured on its ST bins
(clamped to 0.25-0.75). The solids are meshed with the same code as the GC
columns, so the two line up in the viewer. If sections were skipped when
cutting, the gaps are bridged exactly as the columns bridge them.

**IF images.** The red, green and blue bands are the config's display channels
(`if_images.display_rgb`, or the fluorophores in the channel names when that is
unset). Each channel's contrast limits are its 1st and 99.7th percentiles over
all IF sections together, so brightness can be compared between sections. The
GC outlines are traced on the run's GC label rasters, in image pixels.

## Other files in this folder

- `extract_rctd.R` writes one RCTD weight table per ST section (see above):
  `Rscript extract_rctd.R section_rctd.rds out.csv.gz sample_name`.
- `layer_common.py` is shared code for the builders (reading the run, finding
  the RCTD tables).
- `build_rctd_four_slice.py` is not a viewer layer. It builds a separate
  interactive HTML atlas of RCTD cell types on four ST sections, from a
  four-section AnnData file that is not part of a gcalign run. It still names
  its inputs in a `REPS` table with `/path/to/...` placeholders; set them before
  running it.
