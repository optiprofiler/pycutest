# PyCUTEst

This repository provides an interface for OptiProfiler to access the [PyCUTEst](https://github.com/jfowkes/pycutest) problem collection.

It contains adaptation tools allowing OptiProfiler to invoke and work with these problems.

## Contents

- **Adaptation Tools**: Wrapper scripts and utilities in the root directory that bridge OptiProfiler with the PyCUTEst collection.

## Package and Plugin

This package definition is a development build for the API-v1 protocol. The
corresponding OptiProfiler release and this plugin distribution have not been
published yet. Until that release exists, test from checkouts with `--no-deps`
as shown below; the `0.1.0` package value is build metadata, not a release
announcement.

The Python distribution name is `optiprofiler-pycutest`. It installs the
adapter package `optiprofiler_pycutest` and registers the problem-library entry
point

```toml
[project.entry-points."optiprofiler.problem_libraries"]
pycutest = "optiprofiler_pycutest:get_problem_library"
```

The adapter package name intentionally avoids `pycutest`, which is reserved for
the upstream PyCUTEst runtime module. Installing or discovering this plugin does
not import PyCUTEst or compile CUTEst problems. Runtime availability is checked
only when OptiProfiler actually selects the `pycutest` problem library.

For local development against a checked-out OptiProfiler core:

```bash
python -m pip install -e /path/to/optiprofiler
python -m pip install -e . --no-deps --no-build-isolation
```

This distribution intentionally installs only the OptiProfiler adapter. It does
not install or configure PyCUTEst, CUTEst, or MASTSIF. Users who want to run
PyCUTEst problems must prepare that runtime independently by following the
[upstream PyCUTEst installation instructions](https://jfowkes.github.io/pycutest/_build/html/install.html).
OptiProfiler checks runtime availability only when the `pycutest` library is
selected; installing and discovering the adapter remain lightweight.

To remove only this adapter and its OptiProfiler entry point, run:

```bash
python -m pip uninstall optiprofiler-pycutest
```

This does not remove or reconfigure PyCUTEst, CUTEst, MASTSIF, compiled CUTEst
problems, or user caches. Those external runtime components remain under their
own installation and cleanup procedures.

## Discover, Update, and Remove

Discovery reads the installed entry-point metadata without importing the
upstream PyCUTEst runtime:

```python
from optiprofiler import list_problem_libraries

assert "pycutest" in list_problem_libraries()
```

Select the adapter with `benchmark(..., plibs=["pycutest"])`; an installed
plugin needs no filesystem path.

For an unpublished development checkout, update the compatible feature branch
and reinstall the adapter explicitly:

```bash
git pull --ff-only
python -m pip install -e . --no-deps --no-build-isolation
```

The OptiProfiler core repository records the tested adapter commit in
`problem_libraries.lock`. Do not assume that an arbitrary newer adapter commit
is compatible with an older core checkout.

Running `python -m pip uninstall optiprofiler` does not uninstall this adapter
or its upstream runtime. The adapter remains installed but cannot be used until
a compatible OptiProfiler core is installed again. Running
`python -m pip uninstall optiprofiler-pycutest` removes only this adapter and
its entry point; the independently managed runtime and caches are preserved as
described above.

## Configuration

The file `config.txt` in this directory controls how `pycutest_select` filters problems (e.g., `variable_size` and `test_feasibility_problems`). See the comments in `config.txt` for a full description of each option.

For a reproducible OptiProfiler experiment, pass the options explicitly for
this run:

```python
from optiprofiler import benchmark

benchmark(
    solvers,
    plibs=['pycutest'],
    plib_options={
        'pycutest': {
            'variable_size': 'all',
            'test_feasibility_problems': 2,
        },
    },
)
```

OptiProfiler stores the validated effective mapping with the experiment. For a
process-level default shared by subsequent calls, the compatibility API remains
available:

```python
from optiprofiler import set_plib_config, get_plib_config

# View the current effective configuration
print(get_plib_config('pycutest'))

# Override subsequent calls in the current Python process
set_plib_config('pycutest', variable_size='all', test_feasibility_problems=2)
```

The precedence is per-run `plib_options`, process-level `set_plib_config`,
environment variables, `config.txt`, then built-in defaults. You can also set
`PYCUTEST_VARIABLE_SIZE` and `PYCUTEST_TEST_FEASIBILITY_PROBLEMS` directly.
The adapter merges these layers first and validates the final mapping once, so
an explicit valid per-run value can replace an invalid lower-priority value.
Inspecting these options does not import PyCUTEst or require a working CUTEst
installation. Runtime availability is checked only when this library is
selected for a benchmark.

## Testing

The `CI` workflow runs daily and on pushes on Linux. It installs CUTEst/PyCUTEst, checks the OptiProfiler adapter layer, and keeps the sample intentionally small:

- load `ROSENBR` through `pycutest_load` and evaluate `fun`, `cub`, and `ceq`;
- select small problems through `pycutest_select`;
- check `variable_size` and `test_feasibility_problems` environment overrides;
- check the installed entry-point factory without importing PyCUTEst;
- sample a few additional small problems each day with at most two numerical-library threads.

If PyCUTEst is not installed locally, the Python tests are skipped. From this repository:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Maintenance

The `Collect Problem Info` workflow regenerates the metadata used by `pycutest_select`. It is heavier than the smoke CI because it installs CUTEst/PyCUTEst and scans the problem list in blocks.

## Provenance and License

The OptiProfiler adapter code and generated metadata in this repository are
licensed under the [BSD 3-Clause License](LICENSE). The built distribution does
not include PyCUTEst, CUTEst, or MASTSIF source code or compiled libraries.

[PyCUTEst](https://github.com/jfowkes/pycutest) is an optional external runtime
dependency licensed under GPL-3.0-or-later. CUTEst and MASTSIF are installed
separately and retain their own upstream provenance, citation, and license
requirements. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the
component boundaries and upstream links.
