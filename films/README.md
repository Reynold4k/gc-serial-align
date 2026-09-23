# Films

A renderer that turns a finished nine-section run into one continuous film: the
Stereo-seq sections converging, the frame lifting into the cutting axis, each
immunofluorescence section being matched in, and the finished stack turning once.
`gcalign` needs none of this — it is a way of showing what the registration did, built
so that every moving thing on screen is a transform the pipeline actually stored.

| Script | What it does |
|---|---|
| `film_paths.py` | every installation-specific path, in one place; run it to see what still needs setting |
| `story_core.py` | the camera and the drawing primitives — perspective projection, textured section quads, point splatting, HUD |
| `story_assets.py` | reads the pipeline's output and bakes it into the textures each act needs |
| `render_story.py` | the storyboard, the HUD, and the encode; `--preview` renders key frames without encoding |
| `check_film.py` | pulls frames back out of the finished MP4 and tiles them, to check the cut as a whole |
| `make_compact.sh` | a smaller copy of each film for slides — same frames, cheaper rate |
| `run_story.sbatch` | both replicates as a SLURM array, ~40 min each on one core |

## The five acts

| Act | Length | What happens |
|---|---|---|
| 1 | 21.0 s | Four Stereo-seq sections find one common frame. All 1,100 stored Spateo iterations, in their saved order, one iteration per frame. The reference section never moves. |
| 2 | 4.6 s | The same four sections tilt out of the top view and separate along the cutting axis, while the point clouds dissolve into sections with a thickness. |
| 3 | 41.0 s | Five IF sections, 8.2 s each: the section slides in, the camera drops to a top view, the 12 stored Stage-A ICP iterations run, the view pushes into the densest neighbourhood of GCs, the thin-plate spline result takes over, and the section settles into its slot. |
| 4 | 4.0 s | Nine sections in one coordinate system. |
| 5 | 15.2 s | Re-coloured to the viewer's own palette — `#FF1493` Stereo-seq, `#0F52BA` IF — and one full turn. |

## What it reads

Nothing here recomputes a registration.

| File | Used for |
|---|---|
| `02_global_align/global_transforms.json` | the Stage-A similarity and per-iteration ICP history for each IF section |
| `rep{n}_verified_icp_replay.json` | the 12 ICP iterations as transforms, from `scripts/replay_saved_icp.py` |
| `03_gc_match/plane_gc_labels.npz` | the final GC id raster for all nine planes |
| `03_gc_match/if_channels_warped.npz` | the real stain, warped by the same sampling map as the labels |
| `03_gc_match/match_report.json` | anchor counts, TPS residual, leave-one-out error |
| `02_global_align/st_rasters.npz` | the frozen Stereo-seq tissue and GC masks |
| `01_if_gc/IF*_gc.npz` | each IF section still in its own frame, for the ICP replay |
| the upstream Spateo run | `uns/iter_rigid`, `uns/iter_nonrigid`, `uns/iter_deformable` — act 1 only |

`render_story.py` writes a `rep{n}_story_validation.json` next to the film: act boundaries,
iteration counts, the SHA-256 of each input, and the list of approximations below.

## Paths

Everything installation-specific lives in `film_paths.py` as a placeholder with an
environment-variable override. Nothing else in this directory hard-codes a path.

```bash
export GCALIGN_NINE_SLICE_DIR=/path/to/_build/nine_slice   # holds rep1/, rep2/
export GCALIGN_FILM_OUT=/path/to/film_output
export GCALIGN_ICP_REPLAY_DIR=$GCALIGN_FILM_OUT
# act 1 only - the Stereo-seq run that still has its stored iterations
export GCALIGN_SPATEO_REP1=/path/to/rep1_bin20_aligned
export GCALIGN_SPATEO_REP2=/path/to/rep2_bin20_aligned
export GCALIGN_GC_FAMILY_CSV=/path/to/new_region_merged_gc_id.csv
export GCALIGN_GC_REGIONS_DIR=/path/to/02_GC_selection
export GCALIGN_SLICES_REP1=CHIP_A,CHIP_B,CHIP_C,CHIP_D
export GCALIGN_SLICES_REP2=CHIP_E,CHIP_F,CHIP_G,CHIP_H

python film_paths.py                                  # what resolves, what does not
python render_story.py --rep rep1 --preview           # key frames only, no encode
python render_story.py --rep rep1                     # the film, ~40 min
./make_compact.sh rep1 rep2                           # slide-deck copies
python check_film.py --rep rep1                       # 25 frames back out of the MP4
```

Needs `numpy`, `scipy`, `scikit-image`, `pillow`, `h5py`, `pandas`, `pyarrow`, and an
`ffmpeg` with libx264 — on `PATH`, at `GCALIGN_FFMPEG`, or via `imageio-ffmpeg`.

## Three things that are staging, not measurement

All three are captioned on screen and listed in the validation JSON.

**The z axis is exaggerated ×45.** The true section pitch is 10 µm across roughly 8 mm of
tissue; at true scale the nine sections are one smear. The drawn slab is thinner than the
pitch as well, or the stack would be solid.

**Each IF section's sideways entry is choreography.** Where it lands is not: the slide ends
at the real Stage-A placement, which is ICP iteration 0, and the iterations run from there.

**The warped IF outline is re-cut.** The pipeline stores no warped tissue mask, so the
silhouette that gives each IF section a body is re-cut from the warped channels with
`gcalign.if_sections.tissue_mask`'s own recipe. GC identities, fluorescence and every
number on screen are untouched by it.

One quantity-level inconsistency comes from the pipeline itself and is reproduced rather
than silently fixed: the ICP centroids use the manifest's `um_per_px` (9.985 µm) while the
raster warp uses `grid_um` (10.0 µm). That is 0.15 %, about one pixel at film scale.

## Three things worth knowing about how it is drawn

**It is a real camera, not an oblique projection.** `story_core.Camera` is a look-at
pinhole with a field of view, so one continuous move can go from the top view, out into
the stack, down onto a single section and round it. A fixed affine — which is what the
earlier animation used — cannot do that, and its "3-D" is a drawing rather than a space.
Each section is a textured quad mapped through a homography; `local_corners()` composes an
IF section's own affine with the camera's homography, so a mid-ICP state costs exactly one
texture transform, the same as a finished one.

**Act 3 pushes in on purpose.** The first three ICP iterations take the mean paired GC
distance from about 165 µm to about 50 µm, which is obvious at section scale. Everything
after that moves by tens of micrometres — under a pixel from up there. So once the large
move is done the top view pushes into the window holding the most paired GCs (chosen
automatically, ~1,150 µm half-width, about ×3.7), and the refinement and the TPS that
follows it happen at a scale where they can be seen.

**Point clouds get a fixed sub-pixel dither.** bin20 bins sit on a regular lattice;
projected straight onto pixels it beats against the pixel grid and the tissue fills with
moiré. One fixed offset per point scatters it, and being fixed rather than per-frame it
never shimmers.
