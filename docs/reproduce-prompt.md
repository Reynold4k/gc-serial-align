# Reproducing this work

Two ways to use this file: run the pipeline on your own data (top), or rebuild the whole
thing from scratch with an AI coding assistant (bottom).

## A. Run it on your data

1. Register your ST sections to each other first. We used Spateo's `morpho_align`; any
   method works, as long as the result is an AnnData with registered coordinates in
   `obsm` and a per-section label plus a structure identity in `obs`.
2. Copy `configs/rep1_mouse_spleen.yaml`, edit the paths, the section list, the channel
   names and `gc_channel`.
3. `gcalign show --config yours.yaml` — check the derived layout and the z span.
4. `gcalign run --config yours.yaml`.
5. Read `validation_report.json` before you believe any picture. The three numbers that
   matter most: did all sections agree on handedness; how many GCs matched within
   `close_match_um`; and the two-sided disagreement on the interior sections.

## B. Rebuild it from scratch

The prompt below reconstructs this package. It is written to be pasted into a coding
assistant that has access to your data and a shell. Replace the bracketed parts.

---

> I have serial sections of [TISSUE] cut at [N] consecutive sections of [T] µm, with no
> sections discarded. Odd-numbered sections went to immunofluorescence (CZI files at
> [IF PATH], channels [CHANNEL → MARKER MAP]); even-numbered sections went to Stereo-seq
> ([ST PATH]). The [M] Stereo-seq sections are already registered to each other and the
> result, together with curated [STRUCTURE] identities, is in [H5AD PATH] — coordinates in
> `obsm['[COORD KEY]']`, section labels in `obs['[SLICE COL]']`, structure ids in
> `obs['[ID COL]']`.
>
> Build a pipeline that places the IF sections into that same coordinate frame and
> reconstructs each [STRUCTURE] in 3D, then writes a single self-contained interactive HTML
> viewer.
>
> Requirements and constraints:
>
> - **Never move the Stereo-seq coordinates.** Treat them as ground truth; the curated
>   identities must stay valid. The IF sections are fitted into that frame.
> - **Drive the registration with the [STRUCTURE]s themselves**, detected independently in
>   each modality — [IF MARKER] on the IF side, the corresponding transcriptomic label on
>   the ST side. Do not impute a transcriptome onto IF cells, and do not assume any
>   cell-to-cell correspondence between sections.
> - **Resample the IF images onto the ST grid first**, so both modalities share one pixel
>   size for the whole pipeline. The scale factor is known from the metadata — do not fit
>   it beyond a small stretch correction.
> - **Enumerate orientations rather than optimising them.** Both mirror states × rotations
>   in 5° steps × FFT translation. Keep several diverse candidates.
> - **Choose the orientation by structure agreement, not tissue overlap.** Check this
>   explicitly: a roughly convex section overlaps nearly as well mirrored, and tissue
>   overlap will pick the wrong branch. Report both scores so the difference is visible.
> - **Align each interior IF section to both flanking ST sections independently**, blend by
>   true z distance, and keep the disagreement between the two as that section's
>   uncertainty. Flag the first and last sections as one-sided.
> - **Pick the elastic warp's stiffness by leave-one-out cross-validation** over the
>   matched anchors, not by eye.
> - **Smooth the 3D surfaces in the voxel volume, not with a mesh smoother** — a mesh
>   smoother collapses thin columns. Choose the iso-level so the enclosed volume matches
>   the measured voxel volume, and pad by ≥4σ on every axis so the meshes close. Verify
>   watertightness and the volume ratio.
> - Before building anything, **measure whether the z span can support the claim I want to
>   make**. Compare the total stack thickness with the diameter of the structure. Tell me
>   plainly if it cannot, rather than producing a picture that implies it can.
>
> Verify as you go rather than assuming: check the channel identities against the expected
> morphology, check that all sections converge to the same handedness, and drive the
> finished HTML in a headless browser to confirm every control works and that it is
> actually interactive before you hand it over.

---

## What to watch out for

Things that cost us time, in the order we hit them:

1. **Check what your images actually contain before designing around them.** Ours turned
   out to have a fourth channel nobody had mentioned, and a DAPI channel too saturated to
   segment. Both changed the plan.
2. **The SAW output already carries a registered nuclear image in chip coordinates.** If
   you are on Stereo-seq, check `outs/image/*_regist.tif` before building a bridge you do
   not need — its pixel grid is the same as the bin coordinates.
3. **Tissue overlap will choose the wrong mirror.** See above.
4. **A mesh smoother will destroy thin columns.** See above.
5. **Interactive pages must be driven, not eyeballed.** Several real bugs — a volume unit
   wrong by 1000×, a camera that pinned to the pole, a view that overflowed the frame —
   only appeared when the page was actually clicked through. A headless browser makes this
   cheap.
6. **Measure before optimising.** We twice "fixed" performance based on a wrong diagnosis,
   including once from a bug in our own benchmark that silently measured a truncated mesh.
