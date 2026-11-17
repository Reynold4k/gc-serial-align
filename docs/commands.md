# Command reference

Every gcalign command takes `--config my_experiment.yaml`, the file that describes the
experiment ([template](../configs/template.yaml)). Add `--output-dir /path/to/run` to use
a run directory other than the `output_dir` written in the config.

| Command | Purpose | Needs a finished run |
| --- | --- | --- |
| [`gcalign show`](#gcalign-show) | print the section layout derived from the config | no |
| [`gcalign check`](#gcalign-check) | test the input files before a run | no |
| [`gcalign run`](#gcalign-run) | run the registration pipeline | no |
| [`gcalign stages`](#gcalign-stages) | list the pipeline stages | no |
| [`gcalign figures`](#gcalign-figures) | draw the result as PNG figures | yes |
| [`gcalign report`](#gcalign-report) | write one HTML page with quality numbers and figures | yes |
| [`gcalign animate`](#gcalign-animate) | render a film of the registration | yes |
| [`gcalign transfer`](#gcalign-transfer) | read the IF intensity at every ST bin | yes |
| [`gcalign export`](#gcalign-export) | write an IF section as a registered image file | yes |
| [`gcalign map`](#gcalign-map) | convert point coordinates between an IF image and the frame | yes |

Terms: **IF** = immunofluorescence; **ST** = spatial transcriptomics; **GC** = germinal
centre (the landmark structure); **bin** = one ST measurement spot; **frame** = the
coordinate system of the registered ST sections, in micrometres.

The commands in the lower part of the table also work on a run made with gcalign 0.1.

---

## gcalign show

Prints the experiment as gcalign understands it: the sections in cutting order, the depth
(z) of each section, which ST sections each IF section is registered to, and which
channels are displayed as red, green and blue. It reads only the config.

```bash
gcalign show --config my_experiment.yaml
```

## gcalign check

Opens every input file and reports problems that would stop a run. It writes nothing.
Exit status 0 means a run can start.

```bash
gcalign check --config my_experiment.yaml
```

It checks that:

- every IF image exists and has the channels named in the config;
- the pixel size in the config agrees with the pixel size stored in the image file;
- the AnnData file has the coordinate key and the columns named in the config;
- every ST section in the config is present in the AnnData and has at least six
  labelled structures;
- the ST bin size matches `align.grid_um`.

It also compares the depth of the section stack with the typical diameter of the
structures and says when the 3D result will be a slab through each structure and not the
whole structure.

## gcalign run

Runs the pipeline. The seven stages run in this order:

| Stage | What it does | Output folder |
| --- | --- | --- |
| `scaffold` | reads the registered ST coordinates and GC ids | `00_scaffold/` |
| `if` | reads each IF image, finds the tissue outline and the GCs | `01_if_gc/` |
| `align` | finds mirror state, rotation, scale and shift of each IF section | `02_global_align/` |
| `match` | matches GCs between IF and ST, fits the elastic warp, gives each GC one id | `03_gc_match/` |
| `columns` | stacks the outlines of each GC into a 3D surface | `04_columns/` |
| `viewer` | writes the interactive HTML viewer | `05_viewer/` |
| `validate` | writes the quality report and checksums | `validation_report.json` |

```bash
gcalign run --config my_experiment.yaml              # all stages
gcalign run --config my_experiment.yaml --jobs 0     # use every available CPU
gcalign run --config my_experiment.yaml --from match # start at a stage; earlier output must exist
gcalign run --config my_experiment.yaml --only viewer
```

| Option | Meaning |
| --- | --- |
| `--jobs N` | number of processes for the stages `if`, `align` and `match`, which handle each IF section separately. `0` = all CPUs. The result is identical for every value of N. |
| `--from STAGE` | first stage to run |
| `--to STAGE` | last stage to run |
| `--only STAGE` | run one stage |

Running time for nine sections (five IF, four ST): about 5 minutes with one process,
1 to 1.5 minutes with five. Stage `align` takes most of the time.

All output files are described in [outputs.md](outputs.md).

## gcalign stages

Prints the stage names and the commands available after a run.

## gcalign figures

Draws the result of a run as PNG figures on a black background. The figures contain no
titles or captions; [figures.md](figures.md) explains each figure.

```bash
gcalign figures --config my_experiment.yaml
gcalign figures --config my_experiment.yaml --gc 3,12      # also one close-up figure each for GC 3 and GC 12
gcalign figures --config my_experiment.yaml --only sections,overlay --dpi 300
```

| Option | Meaning |
| --- | --- |
| `--only NAMES` | figure names to draw, comma separated (default: all) |
| `--gc IDS` | GC ids for close-up figures, comma separated, or `all` |
| `--dpi N` | resolution of the PNG files (default 220) |
| `--out DIR` | output folder (default `06_figures/` in the run directory) |

## gcalign report

Writes `06_figures/report.html`: one HTML file that contains the figures and the quality
numbers of the run. It opens in any browser without a network connection.

```bash
gcalign report --config my_experiment.yaml
```

The report lists, for each IF section, the chosen orientation, the number of matched GCs,
the registration error before and after the elastic warp, the leave-one-out error and the
two-sided disagreement. It also lists which GC appears on which section, the checks on the
3D surfaces, and the fraction of ST tissue covered by each IF section.

## gcalign animate

Renders a film (MP4 video) that replays the registration from the files of a finished
run. An `ffmpeg` program is required; `pip install "gcalign[film]"` installs one.

```bash
gcalign animate --config my_experiment.yaml                  # the full film
gcalign animate --config my_experiment.yaml --preset short   # a short version
gcalign animate --config my_experiment.yaml --preview        # still images only, no video
```

The film has up to five acts:

| Act | Content | Data shown |
| --- | --- | --- |
| 1 | the ST sections are registered to each other | the stored iterations of the Spateo run (only with `--spateo-dir`) |
| 2 | the registered ST sections are stacked in cutting order | the ST bins and their GC ids |
| 3 | each IF section is matched to its neighbouring ST sections | every ICP iteration stored by stage `align`, then the result of the elastic warp |
| 4 | all sections in one coordinate system | the registered sections |
| 5 | the stack turns once | the GC regions, IF in blue and ST in pink |

ICP = iterative closest point, the step that refines the placement of an IF section on
the GC centres.

| Option | Meaning |
| --- | --- |
| `--preset full\|short` | `full` (default): every stored iteration; about 66 s for five IF sections, 86 s with act 1. `short`: about 17 s (20 s with act 1); one IF section is matched on screen and the others are placed directly |
| `--spateo-dir DIR` | folder of the Spateo run that registered the ST sections. Act 1 needs it. Without it the film starts at act 2 |
| `--no-act1` | leave out act 1 |
| `--label TEXT` | a short label shown at the top right of every frame, for example `rep1` |
| `--size WxH`, `--fps N` | frame size and frame rate (default 1920x1080, 60) |
| `--quality master\|draft` | `draft` renders faster and is less sharp |
| `--workers N` | number of processes that render frames |
| `--preview` | write still images of key moments and no video |
| `--out FILE` | output file (default `07_film/<name>.mp4` in the run directory) |

Rendering time, measured on our data at 1920 × 1080 and 60 frames per second:

| Film | Frames | 16 CPUs | 1 CPU |
| --- | --- | --- | --- |
| full film (`--preset full`) | 5,149 | about 5 minutes | about 45 minutes (estimated from the rate of the short film) |
| short film (`--preset short`) | 1,184 | about 2 minutes | 10 minutes |

A file `<name>_validation.json` is written next to the film. It records the length of
every act, which stored data each act used, and which parts of the film are staged
(for example the camera movement and the stretch of the depth axis) and not measured.
[`films/README.md`](../films/README.md) describes the film in detail.

## gcalign transfer

Reads, for every ST bin, the IF intensity of each channel at the position of that bin on
the neighbouring IF sections. The result links protein signal to transcriptome bins.

```bash
gcalign transfer --config my_experiment.yaml               # writes 06_transfer/st_bins_if.npz
gcalign transfer --config my_experiment.yaml --format csv  # writes a CSV file
```

The output has one row per ST bin, in the row order of the AnnData file. Columns:

| Column | Content |
| --- | --- |
| `obs_names` | the row name of the bin in the AnnData file (written when the AnnData file is found at the path given in the config) |
| `section` | name of the ST section the bin belongs to |
| `x_um`, `y_um` | position of the bin in the frame, micrometres |
| `gc_id` | id of the GC the bin belongs to on its ST section; 0 = not in a GC |
| `if_before`, `if_after` | names of the IF sections cut immediately before and after the ST section |
| `<channel>_before`, `<channel>_after` | intensity of the channel on each of these IF sections at the bin position; NaN if the bin lies outside the tissue of that IF section |
| `<channel>` | mean of the two values, weighted by 1 / distance in z; NaN if neither IF section covers the bin |
| `covered_before`, `covered_after` | true if the bin lies inside the tissue outline of that IF section |
| `if_gc_before`, `if_gc_after` | id of the GC found on that IF section at the bin position; 0 = none; negative = a GC seen only in IF |

`st_bins_if.json`, written next to the table, records the channels, the units, and one
`scale` value per IF section and channel.

How to read the values:

- The value is the stain on a neighbouring section, one section thickness above or below
  the bin. IF and ST are never measured on the same section.
- The unit is the raw image intensity after correction of the stitching stripes.
  Intensities are comparable within one IF section. To compare IF sections, divide by the
  `scale` of that section and channel (the 99.9th percentile of the section).
- Agreement between a GC marker and the ST GC labels is expected, because the
  registration uses the GCs. It confirms that the run is consistent; it is not independent
  evidence that the registration is correct. In our data, GL7 intensity read in this way
  separates GC bins from other bins with an AUC of 0.98 (replicate 1, all four ST
  sections) and 0.96–0.98 (replicate 2).

Attach the table to the AnnData object:

```python
import anndata as ad, numpy as np
adata = ad.read_h5ad("registered_sections.h5ad")
t = np.load("out/06_transfer/st_bins_if.npz")
assert (t["obs_names"] == adata.obs_names).all()
adata.obs["GL7_if"] = t["GL7_AF488"]
```

## gcalign export

Writes one IF section as a multi-channel OME-TIFF image that is registered to the frame,
at the pixel size you choose. The pipeline itself works at 10 µm per pixel; `export`
reads the original image again and applies the same registration at a finer pixel size.

```bash
gcalign export --config my_experiment.yaml --section IF3 --um-per-px 2
gcalign export --config my_experiment.yaml --section all --um-per-px 1 --channels GL7_AF488,IgD_AF647
```

| Option | Meaning |
| --- | --- |
| `--section NAME` | the IF section to export, or `all` |
| `--um-per-px P` | pixel size of the output image in micrometres (default 2) |
| `--channels NAMES` | channels to write, comma separated (default: all) |
| `--no-warp` | apply only the placement (mirror, rotation, scale, shift), not the elastic warp |
| `--out FILE` | output file (default in `06_registered_images/` in the run directory) |

Two files are written per section:

- `<section>_registered_<P>um.ome.tif` — the image, channels × rows × columns, with channel
  names and pixel size in the OME metadata;
- `<section>_registered_<P>um.json` — the position of the image in the frame:
  `x_um = x0_um + column × um_per_px`, `y_um = y0_um + row × um_per_px`. Divide by
  `frame_um_per_anndata_unit` to obtain the coordinate units of the AnnData file.

Notes:

- The exported image is the original image without correction of the stitching stripes.
- A smaller pixel size does not make the registration more accurate. The accuracy is the
  leave-one-out error given in the report (17–47 µm in our data).

## gcalign map

Converts point coordinates between the original IF image and the frame. Use it to place
objects measured on the IF image (for example cell centres from a segmentation) in the
coordinate system of the ST data, or to find ST positions in the IF image.

```bash
# IF image pixels -> frame
gcalign map --config my_experiment.yaml --section IF3 --points cells.csv
# frame -> IF image pixels
gcalign map --config my_experiment.yaml --section IF3 --points bins.csv --from frame
```

The input CSV needs the columns `x` and `y`.

| `--from` | Input `x`, `y` | Columns added to the output |
| --- | --- | --- |
| `image` (default) | pixel coordinates in the original IF image | `x_frame_um`, `y_frame_um` (micrometres in the frame) and `x_anndata`, `y_anndata` (the units of the AnnData coordinates) |
| `frame` | micrometres in the frame | `x_image_px`, `y_image_px` (pixel coordinates in the original IF image) |

The output is written next to the input file as `<input>_<section>_mapped.csv`, or to
`--out FILE`.

---

## Use from Python

The same operations are available as a Python interface.

```python
import gcalign

cfg = gcalign.load("my_experiment.yaml")          # the experiment description
```

**Convert coordinates** (`gcalign.Transforms`):

```python
t = gcalign.Transforms.load(cfg)["IF3"]           # the registration of IF section IF3

frame_um = t.image_px_to_frame(cells_xy_px)       # IF image pixels -> frame micrometres
image_px = t.frame_to_image_px(bins_xy_um)        # frame micrometres -> IF image pixels
shift_um = t.displacement_um(frame_um)            # distance moved by the elastic warp at each point
```

`frame_to_image_px` is the mapping that the pipeline used to produce its own registered
images. `image_px_to_frame` is its numerical inverse. Converting a point there and back
returns the starting position to within 0.001 µm.

**Read the results of a run** (`gcalign.Run`):

```python
run = gcalign.Run(cfg)

run.names                 # section names in cutting order, e.g. ['IF1', 'D1', 'IF3', ...]
run.planes                # per section: name, kind (IF or ST), section number, depth z in µm
run.labels("IF3")         # 2D array: GC id at each pixel of the frame grid (0 = no GC)
run.rgb("IF3")            # 2D colour image of the IF stain, registered
run.tissue("D1")          # 2D mask of the tissue outline
run.grid.extent()         # [left, right, bottom, top] in µm, for matplotlib imshow
run.transforms["IF3"]     # placement of the section and number of matched GCs
run.match["IF3"]          # elastic warp: anchors, errors, cross-validation curve
run.icp("IF3")["steps"]   # each ICP iteration of the placement
run.presence              # table: area of each GC on each section
run.metrics               # table: volume, radius and drift of each 3D column
run.identity              # colour of each GC id, the same in all outputs
```

All arrays returned by `run` use the same pixel grid (the frame at `align.grid_um`
micrometres per pixel), so they can be overlaid directly.
