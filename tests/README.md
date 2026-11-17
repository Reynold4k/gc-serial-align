# Tests

The test suite checks the pipeline and every command. It needs no external data.

```bash
pip install -e ".[test,film]"
pytest
```

Running time: about two minutes on one CPU.

| File | What it tests |
| --- | --- |
| `test_config.py` | reading the config: section layout, depth of each section, error messages for wrong entries |
| `test_theme.py` | the colours and fonts: black background, readable text, stable GC colours, font files present |
| `test_geometry.py` | single functions of the registration (similarity fit, GC matching, cross-correlation, ICP, filling of skipped sections) against known results |
| `test_pipeline_synthetic.py` | the complete pipeline and the commands `check`, `transfer`, `export`, `map` on synthetic data with a known transform |
| `test_viewer.py`, `test_figures.py`, `test_film.py` | the viewer, the figures, the report and the film are produced for the synthetic run |

## The synthetic test data

`synthetic.py` generates a small experiment for which the correct answer is known:

- Structures (ellipses) and a tissue outline are defined mathematically.
- ST sections are produced by sampling the structures on a bin lattice.
- Each IF section is produced as a multi-channel TIFF image, rendered through a known
  mirror state, rotation, scale and shift, plus a smooth elastic distortion of 22 µm.

The tests then require that the pipeline:

- recovers the mirror state and the rotation of every IF section;
- places points of the IF image at their true position in the frame;
- gives every structure one id on all sections;
- measures the volume that the synthetic scene defines;
- writes identical files with one process and with three processes.

The synthetic experiment has five sections, and its section names and channel names
differ from our real data. A second variant has skipped sections. The tests therefore also
show that gcalign does not depend on nine sections, on the names `IF<n>`, or on our
fluorophores.

The synthetic data is used only in tests. It does not appear in any figure or document.

## Comparing two runs

`tests/regression/compare_runs.py` compares two run directories value by value:

```bash
gcalign run --config my_experiment.yaml --output-dir /tmp/new_run
python tests/regression/compare_runs.py /path/to/reference_run /tmp/new_run
```

For every array, table and report value in the reference run, the script checks that the
new run contains the same value. Files and keys that exist only in the new run are listed
and not treated as differences. Exit status 0 means that nothing in the reference run has
changed.

gcalign 0.2.0 was compared with gcalign 0.1.0 in this way on both replicates before
release. All registration results were identical.
