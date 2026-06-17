# PyCUTEst

This repository provides an interface for OptiProfiler to access the [PyCUTEst](https://github.com/jfowkes/pycutest) problem collection.

It contains adaptation tools allowing OptiProfiler to invoke and work with these problems.

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
- sample a few additional small problems each day with at most two numerical-library threads.

If PyCUTEst is not installed locally, the Python tests are skipped. From this repository:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Maintenance

The `Collect Problem Info` workflow regenerates the metadata used by `pycutest_select`. It is heavier than the smoke CI because it installs CUTEst/PyCUTEst and scans the problem list in blocks.

## Provenance and License

This repository provides an OptiProfiler adapter and metadata for [PyCUTEst](https://github.com/jfowkes/pycutest). PyCUTEst is licensed under GPL-3.0. CUTEst and MASTSIF are separate upstream projects with their own citation and license requirements; please follow the upstream guidance when using the collection.
