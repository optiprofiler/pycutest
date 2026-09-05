# PyCUTEst

This repository provides an interface for OptiProfiler to access the [PyCUTEst](https://github.com/jfowkes/pycutest) problem collection.

It contains adaptation tools allowing OptiProfiler to invoke and work with these problems.

## Paper instance consistency

This backport retains the interface, selector, configuration and CSV from
`44906dd8a5ccdf9eb47c9e315fac834f1cfc1c85`. After loading, it checks the dimension
and bound/linear/nonlinear/total constraint counts against an unambiguous
record in that original CSV. Bare names use the default record; explicitly
parameterized names use an exactly matching `argins` record. Unrecorded custom
parameters are passed to PyCUTEst as before, including its own error handling.
Encoded parameters retain the paper API's precedence over keyword parameters.

A mismatch raises `RuntimeError` rather than silently benchmarking an instance
outside the requested range. No default-parameter table is added, no SIF
parameter is injected, and no problem is replaced or removed from selection.
For example, the frozen CSV describes default `OSCIPATH` as dimension 10.
MASTSIF v0.5 (`04280876fc4c7cae9a06ab34e0d5887bee4d8960`) loads dimension 10;
the separately tested MASTSIF snapshot `29adac9f1618cd23b67f16fe80910e7c030a657a`
loads dimension 500 and is rejected. The test environment's exact runtime
versions must accompany an experiment; the adapter does not install runtimes.

The regression test `tests/test_instance_invariant.py` covers the frozen
runtime, explicit/encoded parameters, legacy precedence, unrecorded parameters,
and count mismatches. Set `OP_TEST_DRIFT_RUNTIME=1` only when explicitly testing
the dimension-500 runtime above in a separate process and fresh cache.

## Contents

- **Adaptation Tools**: Wrapper scripts and utilities in the root directory that bridge OptiProfiler with the PyCUTEst collection.

## Configuration

The file `config.txt` in this directory controls how `pycutest_select` filters problems (e.g., `variable_size` and `test_feasibility_problems`). See the comments in `config.txt` for a full description of each option.

When used through **OptiProfiler**, you can override these options at runtime without editing `config.txt`:

```python
from optiprofiler import set_plib_config, get_plib_config

# View the current effective configuration
print(get_plib_config('pycutest'))

# Override at runtime (persists for the current Python process)
set_plib_config('pycutest', variable_size='all', test_feasibility_problems=2)
```

You can also set the environment variables `PYCUTEST_VARIABLE_SIZE` and `PYCUTEST_TEST_FEASIBILITY_PROBLEMS` directly. Environment variables take precedence over `config.txt`.

## Testing

The `CI` workflow runs daily and on pushes on Linux. It installs CUTEst/PyCUTEst, checks the OptiProfiler adapter layer, and keeps the sample intentionally small:

- load `ROSENBR` through `pycutest_load` and evaluate `fun`, `cub`, and `ceq`;
- select small problems through `pycutest_select`;
- check `variable_size` and `test_feasibility_problems` environment overrides;
- sample a few additional small problems each day with at most two numerical-library threads.

If PyCUTEst is not installed locally, the Python tests are skipped. From this repository:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Maintenance

The `Collect Problem Info` workflow regenerates the metadata used by `pycutest_select`. It is heavier than the smoke CI because it installs CUTEst/PyCUTEst and scans the problem list in blocks.

## Provenance and License

This repository provides an OptiProfiler adapter and metadata for [PyCUTEst](https://github.com/jfowkes/pycutest). PyCUTEst is licensed under GPL-3.0. CUTEst and MASTSIF are separate upstream projects with their own citation and license requirements; please follow the upstream guidance when using the collection.
