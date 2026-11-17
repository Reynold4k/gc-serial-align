# Output files

`gcalign run` writes one folder per pipeline stage into the run directory (`output_dir`
in the config). This page lists every file, its format and its content.

Terms: **IF** = immunofluorescence; **ST** = spatial transcriptomics; **GC** = germinal
centre (the landmark structure); **bin** = one ST measurement spot; **frame** = the
coordinate system of the registered ST sections, in micrometres; **common grid** = the
pixel grid of the frame, `align.grid_um` micrometres per pixel (10 µm by default);
**plane** = one section (IF or ST) placed in the frame. `<IF>` stands for the name of an
IF section, `<ST>` for the name of an ST section, `<id>` for a GC id.

All coordinates are in micrometres unless a column name ends in `_px`. In every 2D array
(raster), row 0 is the smallest y and column 0 is the smallest x of the frame.

## Overview

| Path | Content |
| --- | --- |
| `00_scaffold/` | the ST bins as read from the AnnData file |
| `01_if_gc/` | each IF section before registration: tissue outline, GC mask, channels |
| `02_global_align/` | the placement of each IF section (mirror, rotation, scale, shift) |
| `03_gc_match/` | the elastic warp, the registered images and the GC id of every pixel |
| `04_columns/` | the 3D surface of each GC |
| `05_viewer/` | the interactive HTML viewer |
| `validation_report.json` | quality numbers of the whole run |
| `SHA256SUMS.txt` | checksums of the main output files |

Other commands add: `06_figures/` (`gcalign figures`, `gcalign report`), `06_transfer/`
(`gcalign transfer`), `06_registered_images/` (`gcalign export`), `07_film/`
(`gcalign animate`). They are described in [commands.md](commands.md).

## 00_scaffold/ — the ST bins

| File | Format | Content |
| --- | --- | --- |
| `st_scaffold.npz` | NumPy archive, one entry per bin | `slice_short`: section name of the bin · `gc_id`: GC id (0 = not in a GC) · `world_um`: position in the frame (x, y) · `chip_um`: position on the chip before registration (x, y) · `leiden`: cluster label |
| `st_scaffold_summary.json` | JSON | per ST section: number of bins, number of GC bins, number of GCs, extent. Per GC: the centre on each ST section and the mean centre over sections (`consensus_x_um`, `consensus_y_um`) |

The ST coordinates are copied from the AnnData file and are never changed.

## 01_if_gc/ — the IF sections before registration

Each IF image is resampled to the pixel size of the common grid. The arrays in this folder
are in the image's own orientation, not yet in the frame.

| File | Format | Content |
| --- | --- | --- |
| `<IF>_gc.npz` | NumPy archive | `tissue`: mask of the tissue outline · `gc`: mask of all GCs · `gc_labels`: one integer label per connected GC region · one array per channel, named as in the config: intensity after correction of the stitching stripes |
| `<IF>_gc_mask.tif` | 8-bit TIFF | the GC mask (255 = GC), for viewing in an image program |
| `if_gc_manifest.json` | JSON, one entry per IF section | image file, array shape, true pixel size of the resampled image (`um_per_px_x`, `um_per_px_y`), tissue area, number and area of GCs, and for each GC region its area, equivalent radius, centre and eccentricity |

## 02_global_align/ — placement of each IF section

The placement is a similarity transform: `frame_xy = linear · local_xy + offset`, where
`local_xy` is the pixel position in `01_if_gc/<IF>_gc.npz` multiplied by the grid pixel
size.

| File | Format | Content |
| --- | --- | --- |
| `global_transforms.json` | JSON | `grid`: origin (`x0_um`, `y0_um`), size and pixel size of the common grid. `sections.<IF>`: the placement (`flip` = mirrored or not, `angle_deg`, `linear`, `offset`, `total_scale`), the ST sections it was registered to (`neighbours`), tissue and GC overlap after placement (`tissue_dice_final`, `gc_dice_final`), number of GCs found and matched (`n_if_gc`, `n_matched`, `n_matched_under_150um`), median distance between matched GC centres (`match_median_um`), and the matched pairs |
| `orientation_search.json` | JSON | every orientation candidate that was tested in full, per IF section: mirror state, angle, scale, tissue overlap (`tissue_dice`), number of matched GCs (`n_matched`), number within `close_match_um` (`n_close`) and their median distance. `winner` is the index of the candidate that was chosen |
| `icp_trajectory.json` | JSON | the iterative closest point (ICP) refinement of the chosen candidate, per IF section: the transform after each iteration and the GC pairs it was fitted on |
| `st_rasters.npz` | NumPy archive | `tissue_<ST>` and `gc_<ST>`: tissue mask and GC mask of each ST section on the common grid |

## 03_gc_match/ — elastic warp, registered images, GC ids

| File | Format | Content |
| --- | --- | --- |
| `plane_gc_labels.npz` | NumPy archive | one 2D integer array per plane on the common grid: the GC id at each pixel. Positive = GC id from the ST data; negative = a GC found only on that IF section; 0 = no GC. `grid` = `[x0_um, y0_um, height, width]` |
| `if_channels_warped.npz` | NumPy archive | the registered IF images: one 2D array per IF section and channel, key `<IF>\|<channel>`, 16-bit, on the common grid. 65535 corresponds to the 99.9th percentile of that section and channel |
| `if_tissue_warped.npz` | NumPy archive | the registered tissue outline of each IF section (mask on the common grid) |
| `section_transforms.json` | JSON | the complete registration of each IF section: the placement (`linear`, `offset`) and the elastic warp (`tps`: stiffness `smoothing`, anchor positions `source_um` after placement and `target_um` in the frame), plus the image geometry. `gcalign map`, `gcalign export` and `gcalign.Transforms` read this file |
| `gc_presence.csv` | CSV, one row per GC | for each plane the area of the GC in pixels (`<plane>_px`) and µm² (`<plane>_area_um2`); `n_planes`, `n_st_planes`, `n_if_planes`: number of planes on which the GC appears; `planes`: their names |
| `match_report.json` | JSON | per IF section: number of anchors (`n_anchor_pairs`), median distance between matched GC centres before the warp (`centroid_error_pre_um`) and after it (`tps_residual_median_um`), leave-one-out error (`tps_loo_median_um`) and the error for every stiffness tested (`tps_loo_curve`), two-sided disagreement (`two_sided_disagreement`, absent for sections with one ST neighbour), number of GCs seen only in IF (`n_if_only`), IF regions that were split because two ST GCs lay under them (`objects_split_on_st_evidence`) |

## 04_columns/ — 3D surfaces

A column is the 3D surface of one GC, built by stacking its outlines from all planes.

| File | Format | Content |
| --- | --- | --- |
| `gc_columns.npz` | NumPy archive | per GC: `v<id>` vertex positions (x, y, z in µm) and `f<id>` triangles of the surface; `sv<id>`, `sf<id>`: the same surface with fewer triangles; `gc_ids`: the ids that have a surface |
| `gc_plane_patches.npz` | NumPy archive | the flat outline of each GC on each plane, as polygons |
| `gc_column_metrics.csv` | CSV, one row per GC | `volume_um3`: volume measured from the sections (area × section spacing) · `mesh_volume_um3`: volume enclosed by the surface · `mesh_open_edges`: 0 = the surface is closed · `max_equiv_radius_um`, `median_equiv_radius_um`, `radius_cv`: radius of the outlines over the planes and its coefficient of variation · `centroid_drift_um`: largest distance of an outline centre from the mean centre · `z_span_um`: depth range over which the GC is present · `n_planes`, `n_st_planes`, `n_if_planes` |
| `column_build_report.json` | JSON | settings of the surface construction; `plane_slots`: position of each plane in the stack; `bridged_slots`: positions of skipped sections that were filled by interpolation |

A GC gets a surface only if it appears on at least `columns.min_planes` planes (2 by
default).

## 05_viewer/

One HTML file (name set by `viewer.filename`). It contains all data and needs no network
connection. Open it in a browser.

## validation_report.json

Quality numbers of the run, in JSON:

| Key | Content |
| --- | --- |
| `geometry` | number of sections, section thickness, depth of the stack, order of the planes |
| `stage_A_orientation` | per IF section: mirror state, angle, scale. `all_sections_same_handedness`: whether all IF sections have the same mirror state. `angle_spread_deg`: range of the rotation angles |
| `stage_A_gc_match` | per IF section after placement: GCs found, GCs matched, median distance, tissue and GC overlap |
| `stage_B_warp` | per IF section: number of anchors, error before and after the warp, leave-one-out error, two-sided disagreement |
| `stage_C_identity` | number of GCs in the stack, number present on all planes, number present on more than half of the planes, number seen only in ST or only in IF |
| `stage_D_columns` | number of surfaces, number that are closed, ratio of surface volume to measured volume |
| `if_segmentation` | per IF section: tissue area and number of GCs found |
| `if_coverage_of_st_tissue` | fraction of the ST tissue area that each IF section covers |
| `caveats` | the limitations that apply to every run |

"Stage A" is the placement (stage `align`); "stage B" is the elastic warp (stage `match`).

## Differences from gcalign 0.1

- `03_gc_match/gc_presence_9plane.csv` is now `gc_presence.csv`. The content is the same.
- In `validation_report.json`, `n_on_all_9_planes` is now `n_on_all_planes`, and
  `n_on_ge_5_planes` is now `n_on_majority_of_planes`.
- New files: `orientation_search.json`, `icp_trajectory.json`, `section_transforms.json`,
  `if_tissue_warped.npz`.
- All other files have the same content as in 0.1.
