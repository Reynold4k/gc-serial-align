"""The reconstruction film: a finished run drawn back as one continuous shot.

    import gcalign.film
    gcalign.film.run("my_experiment.yaml")                    # acts 2-5, 07_film/<name>.mp4
    gcalign.film.run(cfg, preset="short")                     # an ~18 s edit
    gcalign.film.run(cfg, upstream="/path/to/spateo_run")     # with act 1
    gcalign.film.make_loop("film_short.mp4", "film.webp")     # a README loop

Every moving thing on screen is a transform the pipeline stored; the
validation JSON written next to each film lists what is staging. See
``films/README.md`` in the repository for the acts and the inputs.
"""
__all__ = ["run", "Film", "PRESETS", "QUALITY", "Upstream", "make_loop", "readme_loops",
           "closeup_window", "contact_sheet", "probe", "compact", "find_ffmpeg"]

_WHERE = {"run": "story", "Film": "story", "PRESETS": "story", "QUALITY": "story",
          "Upstream": "upstream", "make_loop": "tools", "readme_loops": "tools",
          "closeup_window": "tools", "contact_sheet": "tools", "probe": "tools",
          "compact": "tools", "find_ffmpeg": "tools"}


def __getattr__(name: str):
    # the renderer is imported on first use, so that `gcalign --help` stays fast
    if name in _WHERE:
        from importlib import import_module
        return getattr(import_module(f".{_WHERE[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
