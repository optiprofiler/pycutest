from .pycutest_tools import (
    pycutest_clear_all_cache,
    pycutest_clear_cache,
    pycutest_check_available,
    pycutest_collect_info,
    pycutest_get_default_options,
    pycutest_get_sif_params,
    pycutest_load,
    pycutest_select,
    pycutest_validate_options,
)


__version__ = '0.1.0'


def _plugin_select(problem_options, library_options):
    return pycutest_select(problem_options, library_options=library_options)


def _plugin_load(problem_name, library_options):
    return pycutest_load(problem_name, library_options=library_options)


def get_problem_library():
    """Return the OptiProfiler problem-library plugin for PyCUTEst."""

    from optiprofiler import ProblemLibraryPlugin

    return ProblemLibraryPlugin(
        name='pycutest',
        api_version=1,
        select=_plugin_select,
        load=_plugin_load,
        collect_info=pycutest_collect_info,
        check_available=pycutest_check_available,
        get_default_options=pycutest_get_default_options,
        validate_options=pycutest_validate_options,
    )


__all__ = [
    'get_problem_library',
    'pycutest_clear_all_cache',
    'pycutest_clear_cache',
    'pycutest_check_available',
    'pycutest_collect_info',
    'pycutest_get_default_options',
    'pycutest_get_sif_params',
    'pycutest_load',
    'pycutest_select',
    'pycutest_validate_options',
]
