# Running gcalign on your own data, and rebuilding it from scratch

This page has two parts. Part A lists the steps to run gcalign on your own sections.
Part B is a prompt for an AI coding assistant that rebuilds the whole pipeline from
nothing; use it to adapt the method to a setup that gcalign does not support.

## A. Run gcalign on your data

1. Register your ST sections to each other. We used `morpho_align` from Spateo; any method
   works. The result must be an AnnData file with the registered coordinates in `obsm`,
   a column in `obs` that names the section of each bin, and a column in `obs` with the
   integer id of the structure each bin belongs to (0 = none).
2. Copy `configs/template.yaml`, which lists every setting with an explanation, or one of
   the two example configs. Edit the paths, the section list, the channel names and
   `gc_channel` (the IF channel that shows your structure).
3. `gcalign show --config yours.yaml` prints the section layout and the depth of the
   stack. Check that it matches your experiment.
4. `gcalign check --config yours.yaml` opens every input file and reports problems.
5. `gcalign run --config yours.yaml --jobs 0` runs the pipeline on all CPUs. The result
   does not depend on the number of CPUs.
6. `gcalign report --config yours.yaml` writes the quality report. Read it before you use
   the result. The three most important values are: whether all IF sections have the same
   mirror state; how many GCs were matched within `close_match_um`; and the two-sided
   disagreement of the IF sections inside the stack.
7. [commands.md](commands.md) describes the commands for figures, the film, the transfer
   of IF intensity to ST bins, and registered images.

## B. Rebuild the pipeline from scratch

The text below is a prompt. Paste it into an AI coding assistant that can read your data
and run commands. Replace the parts in square brackets with your own values.

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

## Problems we met, in the order they occurred

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
