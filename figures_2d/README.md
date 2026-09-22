# Flat figures

A rendered volume is persuasive whether or not the segmentation under it is any
good, so these draw the same GCs flat, one 3x3 grid of the nine sections per GC.

| Script | Figure |
|---|---|
| `plot_gc_closeup.py` | one GC across the nine sections, IF stain in true colour, GC outline and the previous section ghosted |
| `plot_dzlz_split.py` | the same close-up with the dark/light-zone split drawn over the stain it was derived from |
| `plot_dzlz_all.py` | `plot_dzlz_split.py` over every GC of both replicates, plus a manifest |

On the IF panels the stain is left untouched and the two zones are marked by
hatching rather than a colour wash: the question being asked is whether the
split follows CD4, and a solid fill hides exactly that. The hatch *direction*
is what separates them, because over a red-and-green composite the two hatch
colours blend into the stain.

`plot_dzlz_all.py` writes a `manifest.csv` alongside the figures giving, per GC,
how many of the nine planes it appears on and on how many IF planes the median
split could be made at all. Sort by those before looking through them: a GC
present on two planes produces a mostly empty figure, and one too small on every
IF plane produces an outline with no hatching.
