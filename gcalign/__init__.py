"""Align serial immunofluorescence sections into a spatial-transcriptomics frame.

    import gcalign
    cfg = gcalign.load("my_experiment.yaml")
    run = gcalign.Run(cfg)                    # a finished run, opened for reading
    transforms = gcalign.Transforms.load(cfg) # points between an IF image and the frame

The pipeline itself runs from the command line (`gcalign run --config ...`); see
`gcalign.cli` or the project README.
"""
__version__ = "0.2.0"

from .config import Config, load  # noqa: E402

__all__ = ["Config", "load", "Run", "Transforms", "theme", "__version__"]


def __getattr__(name: str):
    # heavier parts are imported on first use, so `import gcalign` stays light
    if name == "Run":
        from .runio import Run
        return Run
    if name == "Transforms":
        from .transforms import Transforms
        return Transforms
    if name == "theme":
        from importlib import import_module
        return import_module(".theme", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
