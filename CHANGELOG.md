# Changelog

## 0.2.0

### Registration results are unchanged

On both replicates, gcalign 0.2.0 writes the same arrays, tables and transforms as
gcalign 0.1.0. This was checked file by file with `tests/regression/compare_runs.py`.

### New commands

| Command | Purpose |
| --- | --- |
| `gcalign check` | tests the input files before a run and reports what would stop the run |
| `gcalign figures` | draws the result as PNG figures on a black background, without captions |
| `gcalign report` | writes one HTML page with the quality numbers and the figures |
| `gcalign animate` | renders a film of the registration from the files of a finished run |
| `gcalign transfer` | reads the IF intensity of each channel at every ST bin |
| `gcalign export` | writes an IF section as a registered OME-TIFF image at a chosen pixel size |
| `gcalign map` | converts point coordinates between an IF image and the registered frame |
| `gcalign run --jobs N` | processes the IF sections in parallel; the result is identical for every N |

### Changes to the pipeline

- Any number of sections and any section names are supported. Version 0.1 required nine
  sections and IF sections named `IF<n>`.
- Skipped sections are placed at their true depth. The 3D surfaces fill each skipped
  position by interpolation, and GC volumes include the skipped depth.
- Multi-channel TIFF and OME-TIFF images can be used as IF input, in addition to CZI.
- A run stops with an explanation when fewer than six structures can be matched on an IF
  section.
- A run stores four additional files: `orientation_search.json` (every orientation
  candidate with its scores), `icp_trajectory.json` (every ICP iteration),
  `section_transforms.json` (the full registration of each IF section) and
  `if_tissue_warped.npz` (the registered tissue outline of each IF section).
- New Python interface: `gcalign.Run` reads the results of a run, and `gcalign.Transforms`
  converts coordinates between an IF image and the frame. Both work on runs made with 0.1.

### One colour scheme and one font for all output

- `gcalign/theme.py` defines the colours and the font for the viewer, the figures, the
  report and the film: black background, grey text, blue for IF, pink for ST, white where
  IF and ST overlap.
- The Inter font is included in the package (SIL Open Font License).
- Each GC id has one colour in all outputs. The default palette is `aurora`. Set
  `viewer.palette: hsv` to use the palette of version 0.1.

### The viewer

The interactive viewer was rewritten. It shows the same data as in 0.1 and adds:

- a table with the registration quality of each IF section (key `Q`);
- a card for the selected GC with its volume, radius, drift and its cross-section area on
  every section;
- search and sorting of the GC list, and selection with the arrow keys;
- a scale bar and a key of the colours of the current view;
- saving the 3D scene as a PNG image (key `S`), automatic rotation (key `R`) and a mode
  that hides the interface (key `H`);
- the address of the page records the current view, so a copied address reopens the same
  view.

[docs/viewer.md](docs/viewer.md) describes every control.

### Tests

- `tests/` holds 85 tests (pytest). They run the whole pipeline on a synthetic experiment
  whose true registration is known, and test the figures, the report, the film and the
  viewer. [tests/README.md](tests/README.md) explains how to run them.
- `tests/regression/compare_runs.py` compares two run folders file by file.
- A GitHub Actions workflow runs the tests on Python 3.9, 3.11 and 3.13.

### Compatibility with current library versions

- gcalign runs without deprecation warnings on scikit-image 0.26, PyVista 0.49 and
  Pillow 13. The functions that these versions deprecate were replaced by equivalent code
  (`gcalign/masks.py` for the two scikit-image functions). The results are identical.

### Bugs fixed

- `validation_report.json` always stated that the rotation angles of all IF sections
  agreed within 3 degrees, whatever the data showed. The statement is now computed from
  the run. For our second replicate the correct value is 26 degrees.
- Stage `columns` stopped with an error when no GC appeared on enough sections for a 3D
  surface. The run now finishes and reports that no surface was built.

### Example configs

- The example configs no longer contain paths of our cluster.
- `configs/rep2_mouse_spleen.yaml` now uses the same ST colour as replicate 1
  (`viewer.st_hex: "#FF1493"`). Version 0.1 used `#FF4500` for replicate 2.

### Renamed files and keys

| 0.1 | 0.2 |
| --- | --- |
| `03_gc_match/gc_presence_9plane.csv` | `03_gc_match/gc_presence.csv` (same content) |
| `n_on_all_9_planes` in `validation_report.json` | `n_on_all_planes` |
| `n_on_ge_5_planes` in `validation_report.json` | `n_on_majority_of_planes` |
| scripts in `films/` | the subpackage `gcalign.film`, used by `gcalign animate` |

The scripts in `figures_2d/` and `viewer_layers/` now take `--config` and no longer
contain replicate names or paths.

## 0.1.0

First release: the seven-stage pipeline, the HTML viewer and two example configs.
