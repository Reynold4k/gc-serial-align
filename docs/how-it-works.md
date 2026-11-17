# Implementation details

This page describes how each pipeline stage works, which parameters control it, and which
problems we met. For the idea behind the method, read [method.md](method.md) first.

Terms: **IF** = immunofluorescence; **ST** = spatial transcriptomics; **GC** = germinal
centre (the landmark structure); **bin** = one ST measurement spot; **frame** = the
coordinate system of the registered ST sections, in micrometres; **common grid** = the
pixel grid of the frame (`align.grid_um`, 10 µm per pixel by default); **anchor** = a GC
matched between an IF section and an ST section.

## The seven stages

| Stage | What it does |
| --- | --- |
| `scaffold` | Reads the registered ST coordinates and the GC id of each bin. It never changes them. |
| `if` | Reads each IF image onto the common grid, corrects the stitching stripes, finds the tissue outline and the GCs. |
| `align` | Finds mirror state, rotation, scale and shift of each IF section (the placement). |
| `match` | Matches GCs between IF and ST, fits the elastic warp, and gives each GC one id on all sections. |
| `columns` | Stacks the outlines of each GC into a closed 3D surface. |
| `viewer` | Writes the interactive HTML viewer. |
| `validate` | Writes the quality report and the checksums. |

The stages `if`, `align` and `match` process each IF section independently. With
`gcalign run --jobs N` each IF section runs in its own process. Stage `align` takes most
of the time: for nine sections, 250 s with one process and 47 s with five. Both runs wrote
identical files.

`gcalign check` tests the inputs before a run. It reports a missing image, a channel that
the file does not contain, a pixel size that disagrees with the image metadata, a section
that is absent from the AnnData file, and a bin size that differs from the grid.

## One pixel grid for both modalities

Stage `if` resamples every IF image to the bin size of the ST data (10 µm by default).
From then on IF and ST use the same pixel size.

The scale between IF and ST is therefore not fitted freely. It follows from the pixel
size of the microscope and the bin size of the chip, which are both known. Stage `align`
fits rotation, mirror state and shift, and allows a scale correction of ±6% for tissue
stretch.

Supported image formats: Zeiss CZI mosaics, and multi-channel TIFF files (plain or OME,
with or without a resolution pyramid). For a pyramidal TIFF, gcalign reads the pyramid
level closest to the grid pixel size and not the full-resolution image.

## Stage `if`: tissue outline and GC segmentation

Two corrections are applied before segmentation:

- **Stitching stripes.** A tiled acquisition leaves a periodic brightness pattern (period
  about 774 µm in our images, 13–20% of the background). gcalign divides it out along both
  image axes, using the median brightness of tissue pixels in each row and column.
- **Tissue outline.** gcalign blends the channels listed in `tissue_channels`, smooths the
  blend, applies an Otsu threshold, closes small gaps, fills holes and keeps the largest
  connected region. If a staining gradient removes part of a section, lower
  `tissue_threshold_scale` (for example to 0.6).

GCs are segmented on the channel named in `gc_channel` with a two-level threshold. A
strict threshold (Otsu) finds GC cores. A lower threshold defines candidate regions. A
candidate region is kept only if it contains a core. This detects dim GCs without
accepting background regions.

Note on DAPI in our data: the DAPI brightness differed more than 13-fold between sections
and was saturated on three of five sections. We therefore did not use DAPI for
segmentation or registration. The marker channels were stable across sections.

## Stage `align`: why the orientation is chosen by GC agreement

Tissue overlap alone selects the wrong mirror state when the tissue outline is nearly
convex. Example (one IF section of replicate 1):

| Candidate | Tissue overlap (Dice) | GC centres within 150 µm |
| --- | --- | --- |
| correct: not mirrored, 355° | 0.9285 | 37 |
| wrong: mirrored, 316° | 0.9291 | 6 |
| wrong: not mirrored, 258° | 0.9301 | 4 |

Two wrong candidates have a higher tissue overlap than the correct one. On a second
section of the same replicate a mirrored candidate scored 0.906 and the correct candidate
0.905, with 11 and 39 GC centres within 150 µm.

gcalign therefore proceeds as follows:

1. Test both mirror states × every rotation in steps of `coarse_angle_step_deg` (5°). For
   each, find the best shift by Fourier cross-correlation of the tissue masks.
2. Keep the `n_orientation_candidates` (6) best candidates that differ from each other.
3. Refine rotation (±6°) and scale (`scale_search`) of each candidate.
4. Run iterative closest point (ICP) on the GC centres of each candidate: up to 12
   iterations, each matching GC centres one-to-one within `icp_match_radius_um` and
   fitting a similarity transform.
5. Choose the candidate with the most matched GC centres within `close_match_um` (150 µm).

Every candidate and both of its scores are stored in
`02_global_align/orientation_search.json`. The `orientation` figure of `gcalign figures`
plots them.

## Stage `match`: stiffness of the elastic warp

The elastic warp is a thin-plate spline fitted through the anchors. Its stiffness
(`smoothing`) is chosen by leave-one-out cross-validation over the values in
`tps_smoothing_grid`: for each value, each anchor is removed in turn, the spline is fitted
on the remaining anchors, and the distance between the predicted and the true position of
the removed anchor is recorded. The value with the smallest median distance is used.

A spline that is too flexible reduces the fitted error to zero and predicts removed
anchors badly. Cross-validation detects this. In our data it mostly selected stiff
settings (10⁵–10⁶). The leave-one-out error was 17–29 µm in replicate 1 and 18–47 µm in
replicate 2.

An anchor is used only if both neighbouring ST sections assign the IF GC to the same GC
id. Stage `match` stops with an error message if an IF section has fewer than six anchors.

## Stage `match`: two GCs that touch on one IF section

Segmentation cannot separate two GCs that touch on one IF section. They form one region.
Without correction, only one of the two GCs can be matched: one GC seems to disappear on
that section and the other seems to grow.

gcalign corrects this with the ST data. An IF region is split when two or more ST GCs each
cover at least `split_min_frac` (6%) of the region and at least `split_min_px` (30)
pixels. The region is divided by a watershed on its distance transform, seeded with the ST
GC regions. A single elongated GC is not split, because only one ST GC lies under it.

In our two replicates the split was applied twice, on two sections of one replicate and
both times for the same pair of GCs. `validation_report.json` lists each split under
`objects_split_on_st_evidence` with the pixel counts before and after.

## Files that record intermediate steps

A run stores the intermediate results of the registration, so that the result can be
checked and reused:

| File | Content |
| --- | --- |
| `02_global_align/orientation_search.json` | every orientation candidate with its tissue overlap and its GC agreement |
| `02_global_align/icp_trajectory.json` | the transform after each ICP iteration and the GC pairs used |
| `03_gc_match/section_transforms.json` | the full registration of each IF section: placement, anchors and stiffness of the warp |
| `03_gc_match/if_tissue_warped.npz` | the tissue outline of each IF section after registration |

`gcalign map`, `gcalign export` and `gcalign.Transforms` read `section_transforms.json`.
`gcalign animate` reads `icp_trajectory.json`.

A run made with gcalign 0.1 does not contain these files. gcalign then recomputes the ICP
iterations and the anchors from the other files of that run, with the same code, and
accepts the result only if it reproduces the transform and the errors stored in that run.

## Stage `columns`: from outlines to a 3D surface

Stacking the outlines of a GC directly gives a stepped shape. gcalign produces a smooth
closed surface in two steps:

1. **Blur the volume.** The stacked outlines are oversampled along z
   (`z_oversample`) and blurred with a Gaussian (`sigma_xy_px`, `sigma_z_sub`). The surface
   is drawn through the blurred volume by marching cubes.
2. **Choose the surface level by volume.** The level is set so that the enclosed volume
   equals the volume measured from the sections. Smoothing changes the appearance of the
   shape and not its volume.

Two problems we met:

- **Mesh smoothing collapses thin columns.** Taubin or Laplacian smoothing of the mesh
  reduced our columns to 1–15% of their volume. gcalign smooths the volume, never the mesh.
- **The volume must be padded.** Without padding of at least 4 standard deviations of the
  blur on every side, the surface is cut at the array border and is not closed. Before
  padding, each of our meshes had about 100 open edges.

## Skipped sections

The `index` of a section in the config is its number in cutting order. Depth follows the
index: with sections 1, 2, 4, 6 and 7 at 10 µm per section, the stack is 60 µm deep.

- Every plane is placed at its true depth in all outputs.
- For the 3D surface, each skipped position is filled by interpolating the signed distance
  to the outline between the two neighbouring measured planes.
- The volume of a GC includes the depth of the skipped positions.
- A position is not filled if the GC is absent on one of the two neighbouring planes.
- `04_columns/column_build_report.json` lists the filled positions (`bridged_slots`).

## The quality report

`validation_report.json` contains, per IF section: the chosen orientation; whether all IF
sections have the same mirror state, and the range of their rotation angles; the number of
matched GCs and their distances; the leave-one-out error; and the two-sided disagreement.
Per GC it contains volume, radius, variation of the radius and drift of the centre across
the stack. It also contains the checks on the 3D surfaces and the general limitations.
[outputs.md](outputs.md) lists every key.

IF sections cut from one block and mounted the same way have the same mirror state. If one
section differs, the report names it. Check how that section was mounted.

## IF coverage of the ST tissue

A GC can be matched on an IF section only where that section has tissue. If an IF section
is torn, folded or imaged over a smaller area than the ST chip, the GCs in the missing
area have no IF counterpart on that section. In the 3D result these GCs are absent on that
plane.

`validation_report.json` reports, for each IF section, the fraction of the ST tissue area
that it covers (`if_coverage_of_st_tissue`). In our two replicates, 68% and 75% of the ST
tissue was covered by all five IF sections. About a quarter of the GCs that were present on
every ST section were missing on at least one IF section for this reason.

Lowering `tissue_threshold_scale` recovers dim tissue. It cannot recover tissue that is
absent. To tell the two cases apart, lower the threshold and count the GC regions found:
when coverage increased from 75% to 83% in our data, the number of GC regions changed
from 278 to 277. The added area contained no GCs, so the missing GCs were caused by the
sections and not by the threshold.

## Test with a known transform

Real sections have no ground truth. gcalign is therefore also tested on synthetic data
with a known answer (`tests/synthetic.py`): structures are defined mathematically, ST
sections sample them on a bin lattice, and each IF section is rendered through a known
mirror state, rotation, scale and shift plus a smooth elastic distortion of 22 µm.

Results of the pipeline on this data:

- The mirror state of every IF section is recovered, including a mirrored section.
- The rotation is recovered to within 0.05°.
- A point of the IF image is placed in the frame with a median error of 1–3 µm and a 95th
  percentile of 12–13 µm. With the placement alone (no elastic warp) the median error is
  23 µm.

The synthetic stain is clean, so these errors are a lower bound for the method. For real
data, use the leave-one-out error in the report. The test suite runs on every change. It
also verifies that a run with several processes writes the same bytes as a run with one.
