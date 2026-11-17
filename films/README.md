# The reconstruction film

`gcalign.film` renders a finished gcalign run as one continuous film (an MP4
video). The film shows how the run placed every section of the tissue block
in one coordinate frame. Every moving thing in the film is a transform that
the run stored. The film does not recompute any registration.

Abbreviations used in this file and in the film:

- IF: immunofluorescence
- ST: spatial transcriptomics
- GC: germinal centre
- ICP: iterative closest point, the point-matching step that places each IF section
- TPS: thin-plate spline, the smooth warp applied to each IF section after the ICP step

## How to render

Render the film of a finished run with the gcalign command line:

```bash
gcalign animate --config my_experiment.yaml                  # writes <output_dir>/07_film/<name>.mp4
gcalign animate --config my_experiment.yaml --preset short   # a 17-20 s edit
gcalign animate --config my_experiment.yaml --preview        # key still images, no video
```

The same renderer runs from Python:

```python
import gcalign.film
gcalign.film.run("my_experiment.yaml")                          # the full film
gcalign.film.run("my_experiment.yaml", preset="short")          # the short edit
gcalign.film.run("my_experiment.yaml", "film.mp4", upstream="/path/to/spateo_run", label="rep1")
```

`python -m gcalign.film render --config my_experiment.yaml` is a third way to
start the same renderer. The command lists its options with `--help`.

Act 1 needs the Spateo run that aligned the ST sections. Give the folder of
the Spateo run with `--spateo-dir` or with the environment variable
`GCALIGN_SPATEO_DIR`. Without the Spateo run, the film starts at act 2.

The film adapts to the run. Section names, section kinds, the number of
sections and their depths (`plane.z_um`) come from the run. The stain colours
come from `display_rgb` in the config. Each IF section gets one matching
sequence, with as many ICP iterations as that section ran.

## What each act shows

Lengths are for a run with five IF sections, rendered at 60 frames per second.

**Act 1: Spateo aligns the ST sections.** Full film 21.0 s, short edit 4.0 s.
The ST sections converge in a top view. The full film shows every stored
Spateo iteration, one per frame (1,100 iterations for the runs this package
was built on). The short edit shows 180 of the stored iterations, evenly
spaced. The reference ST section, the section that Spateo aligned the other
ST sections to, does not move. Stored data:
`uns/iter_<stage>/align_<stage>/<i>` in each Spateo file, joined to the run's
`00_scaffold/st_scaffold.npz` by chip coordinate.

**Act 2: ST sections stacked in cutting order.** Full film 4.6 s, short edit
2.3 s. The aligned ST sections tilt out of the top view and separate along
the cutting axis. The ST bins, drawn as points, become sections with a
thickness. When act 1 is not shown, the film opens on these aligned bins with
a fade-in of 1.4 s (1.0 s in the short edit). Stored data: `00_scaffold/st_scaffold.npz`, `02_global_align/st_rasters.npz`,
`03_gc_match/plane_gc_labels.npz`.

**Act 3: IF sections matched to the ST sections.** Full film 8.2 s per IF
section. Short edit 4.65 s for one IF section plus 0.62 s for each other IF
section. Each IF section slides in to its stored Stage-A placement: the
rotation, mirror state, scale and position that the pipeline's `align` stage
found before the ICP step. The camera drops to a top view of the IF section and
its flanking ST sections, the ST sections directly before and after the IF
section in cutting order. The stored
ICP iterations run one after another, with each matched pair of GC centres
joined by a line. After the first large move, the view pushes in on the
window with the most matched GCs. The TPS result then replaces the last ICP
position, and the section settles into its slot. The short edit matches one
IF section in full: among the IF sections with an ST section on both sides,
the section with the most matched pairs. The other IF sections drop into their
slots already registered. Stored data:
`01_if_gc/<IF>_gc.npz` (the IF section before the match), the ICP steps,
`03_gc_match/plane_gc_labels.npz`, `03_gc_match/if_channels_warped.npz`,
`03_gc_match/match_report.json`.

**Act 4: All sections in one coordinate frame.** Full film 4.0 s, short edit
1.3 s. The finished stack, with the camera swaying slightly. Stored data: the
same files as act 3, showing each IF section after the TPS warp.

**Act 5: One full turn of the stack.** Full film 15.2 s, short edit 5.0 s.
The stack turns 360 degrees. Each GC takes a shade of its section kind's
colour from the 3D viewer: `viewer.st_hex` for ST sections and
`viewer.if_hex` for IF sections. Stored data: the same files as act 4.

With five IF sections, the full film lasts 85.8 s with act 1 and 66.2 s
without act 1. The short edit lasts 19.7 s with act 1 and 16.7 s without
act 1. Act 3 grows with the number of IF sections. Beyond five IF sections,
each matching sequence is shortened so that act 3 stays near 41 s.
No matching sequence is shortened below 60 % of its normal length. A section
whose ICP stopped early gets a shorter ICP phase. A section with no ICP steps
lands on its Stage-A placement and goes straight to the TPS result.

## What appears on screen

- Top left: the act title, for example "Spateo aligns the spatial transcriptomics (ST) sections".
- Top right: the run label given with `--label`, and "z stretched ×40" whenever depth is drawn.
- Bottom left: one counter with its name and total: "Spateo iteration 412 / 1100", "Sections placed 4 / 9", "ICP iteration 5 / 12" or "Rotation 214° / 360°".
- Bottom right: keys for every colour that encodes something: "IF immunofluorescence", "ST spatial transcriptomics", the three stain channels by name, and "one colour per germinal centre (GC)". Top views also show a scale bar.
- Beside the stack: each section's name, in blue for IF sections and pink for ST sections.
- During each match, at the right: the IF section's name and the ST sections it is matched to; a curve of the mean distance between matched GCs per ICP iteration; "matched GC pairs", "mean distance between matched GCs" and "median distance after thin-plate spline"; and a key for the markers (IF GC centre, ST GC centre, matched pair).
- At the foot: a progress line with one segment per act.

## Input files

The film reads the run through `gcalign.runio.Run`.

| Run file | What the film takes from the file |
|---|---|
| `03_gc_match/plane_gc_labels.npz` | the GC id of every pixel of every section |
| `03_gc_match/if_channels_warped.npz` | the IF stain, shown as red, green and blue per `display_rgb`, with the run's shared display limits (`run.rgb`) |
| `03_gc_match/if_tissue_warped.npz` | the outline of each IF section after the warp (runs made with gcalign 0.2 or later) |
| `03_gc_match/match_report.json` | which IF object belongs to which GC, and the median distance after the TPS warp |
| `02_global_align/st_rasters.npz` | the outline of each ST section |
| `02_global_align/icp_trajectory.json` | the stored ICP iterations (runs made with gcalign 0.2 or later) |
| `01_if_gc/<IF>_gc.npz` | each IF section in its own frame, before the match |
| `00_scaffold/st_scaffold.npz` | the aligned ST bins and the GC id of each bin |

Runs made before gcalign 0.2 stored neither the ICP iterations nor the warped
IF outline. For such a run, `gcalign.replay` rebuilds the ICP iterations from
the stored placement. The rebuilt iterations are used only if they end on the
stored transform and reproduce the stored per-iteration history. For such a
run, the IF outline is re-cut from the warped stain (see "What is staged and
not measured").

Act 1 reads one Spateo file per ST section, named
`<source>_gc_bin20_aligned_all_stages.h5ad` (`source` is the ST section's
`source` in the config). Set another name pattern with
`GCALIGN_SPATEO_PATTERN`; `{source}` and `{name}` are filled in. Each Spateo
file supplies `obsm/spatial` (the chip coordinates before alignment) and
`uns/iter_<stage>/align_<stage>/<i>` (the coordinates at each stored
iteration). The GC id of each bin comes from the run's scaffold, matched by
chip coordinate. The renderer refuses act 1 if a scaffold bin is missing from
the Spateo file, or if the final Spateo coordinates differ from the scaffold's
coordinates by more than 1 µm. For both replicates of this project, the final
coordinates agree to 0.0 µm.

## Output files

- `<name>.mp4`: the film. H.264 video in yuv420p with BT.709 colour. The `master` quality uses x264 preset `slow` at CRF 16.
- `<name>_validation.json`: the record of how the film was made. The record lists the frame count and duration, the start and end of every act and of every matching sequence, the ICP iterations of each IF section with their source (stored or rebuilt and verified), the matched GC pairs and mean distance at each iteration, how each IF outline was obtained, the act 1 join, the z stretch factor, a SHA-256 hash of every run file read, and the list of what is staged and not measured.
- `<name>_previews/`: key still images and `storyboard.json`, written by `--preview`.

## What is staged and not measured

The validation JSON next to each film lists the items that apply to that film.

- **The z axis is stretched.** The factor appears on screen, for example "z stretched ×40". By default, the factor makes the stack about 0.44 times as tall as the tissue is wide. Set the factor with `--z-exaggeration`. Each section is drawn at 17 % of its true thickness, so that every section stays visible.
- **The entry of each IF section is choreography.** Each IF section slides in from the right along an arc. The slide ends on the stored Stage-A placement, which is ICP iteration 0.
- **Between two stored ICP iterations, the movement is a blend.** The IF section moves along a straight-line blend of the two stored transforms for a fraction of a second. Every position at which the section comes to rest is a stored ICP iteration.
- **The TPS result replaces the last ICP position by a cross-fade.** The TPS warp itself is not interpolated.
- **During a match, the flanking ST sections are drawn in the plane of the IF section.** The flanking ST sections appear as GC outlines. The other sections fade out during the match.
- **All camera moves are staging.** The camera moves are the dive into the top view, the push-in on the window with the most matched GCs, the sway over the finished stack and the closing orbit.
- **The colours of act 5 are shades of the section kind's colour.** The shades separate neighbouring GCs and carry no other data.
- **ST bins are drawn with a fixed offset of at most half a pixel.** The offset prevents moiré between the regular bin lattice and the pixel grid. Acts 1 and 2 draw every GC bin and every fifth other bin.
- **Runs made before gcalign 0.2 only: the IF outline is re-cut.** The outline is cut from the warped stain with the segmentation's tissue recipe, because such runs stored no warped outline. GC identities, the stain and every number on screen do not depend on the outline. Runs made with gcalign 0.2 or later show the stored warped outline.
- **The short edit only: one IF section is matched on screen.** The other IF sections drop into their slots already registered. The drop along z is staging. Act 1 shows an evenly spaced sample of the stored Spateo iterations; each frame still shows one stored iteration.

## Tools for checking and sharing the film

| Command | What the command does |
|---|---|
| `python -m gcalign.film check FILM.mp4` (or `films/check_film.py`) | reads the duration, frame count and frame rate from the video file, compares the frame count with the validation JSON, and writes a 5 × 5 contact sheet of frames taken from the video |
| `python -m gcalign.film loop FILM.mp4 OUT.webp` | writes a looping animated WebP or GIF of at most 5 MB, for a README (options `--start`, `--duration`, `--width`, `--fps`, `--max-mb`) |
| `python -m gcalign.film loops` (or `films/make_loops.py`) | writes the README loops: each short edit as one loop, and the top-view matching close-up (`gc_matching_topview`) cut from a full film at the times recorded in its validation JSON |
| `python -m gcalign.film compact FILM.mp4 OUT.mp4` | writes the same frames at a lower bit rate (x264 CRF 26; set another value with `--crf`), for slides. For our two full films the copy is about a quarter of the size (217 MB to 58 MB, 178 MB to 49 MB) |
| `films/run_story.sbatch` | a SLURM template that renders the full film and the short edit of each run, checks both, and writes the loops |

The renderer finds ffmpeg in this order: the `GCALIGN_FFMPEG` environment
variable, then `ffmpeg` on `PATH`, then the binary of the `imageio-ffmpeg`
package (the `film` extra). Frames are rendered by one process per core of the
allocation, minus two cores for the encoder (`--workers` sets the number). On
16 cores, the full film renders in about 5 minutes and the short edit in about
2 minutes. On one core, the short edit renders in 10 minutes, which is about two
frames per second. At that rate the full film takes about 45 minutes on one core
(estimated, not measured).

## How the film is drawn

**A pinhole camera.** `gcalign.film.core.Camera` is a perspective camera
with a position, a target and a field of view. One continuous camera move can
therefore go from the top view, out into the stack, down onto one section and
round the stack. Each section is a flat textured rectangle, projected onto
the screen by a perspective transform (a homography). During the ICP
iterations, an IF section is drawn from its texture before the match, placed
by the stored similarity transform of the current iteration.

**The push-in during each match.** The first ICP iterations move an IF
section by one or two hundred micrometres. Later iterations move the section
by a few micrometres, less than a pixel at full-section scale. After the first
large move, the camera therefore pushes in on the window with the most matched
GCs, 0.28 times the tissue's half-width (about 3.6 times closer). The
refinement and the TPS warp then happen at a visible scale.

**Clean edges at every scale.** Section textures are built at three times the
grid resolution, with anti-aliased outlines. Each texture is also stored in
copies filtered by 1, 2 or 4 along each axis (a rip-map). A section seen at a
low angle is sampled from the copy that matches its on-screen size, so
outlines stay continuous and the stain does not shimmer while the camera
moves. Lines and markers are drawn at twice the frame size and filtered down.

**Colour.** The background is pure black (`theme.BG`). IF is blue
(`theme.IF`) and ST is pink (`theme.ST`). Text and lines are neutral grey.
Each GC keeps the run's identity colour (`run.identity`) in acts 1 to 4.
