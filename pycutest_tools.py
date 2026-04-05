import sys, os, io, re, importlib, shutil
from contextlib import redirect_stdout
import numpy as np
import pandas as pd

import builtins
builtins.np = np  # Make numpy available globally as 'np'

# Set the destination directory for pycutest cache.
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, '__pycutest_cache__')
os.makedirs(cache_dir, exist_ok=True)
if cache_dir not in sys.path:
    sys.path.append(cache_dir)
os.environ['PYCUTEST_CACHE'] = cache_dir
import pycutest

# Import Problem class from optiprofiler
from optiprofiler.opclasses import Problem

def pycutest_load(problem_name, **kwargs):
    """
    Convert a PyCUTEst problem name to a `Problem` instance.

    PyCUTEst is a Python interface to the CUTEst optimization test
    environment. More details can be found at
    `the PyCUTEst documentation <https://jfowkes.github.io/pycutest/>`_.

    .. note::

        PyCUTEst is only available on Linux and macOS.

    Parameters
    ----------
    problem_name : str
        Name of the problem in PyCUTEst. You may use `pycutest_select`
        to get the problem names that satisfy your criteria.

    Returns
    -------
    optiprofiler.Problem
        A ``Problem`` instance corresponding to the named problem.

    Notes
    -----
    The problem name may include SIF parameters in the format
    ``'PROBLEMNAME_paramname_paramvalue'`` (e.g., ``'ARGLINA_N_100'``).
    In this case, the function will parse the parameters and pass them
    to PyCUTEst automatically.

    See Also
    --------
    pycutest_select : Select problems from PyCUTEst by criteria.

    Examples
    --------
    .. code-block:: python

        from optiprofiler.problem_libs.pycutest import pycutest_load

        problem = pycutest_load('ROSENBR')
        print(problem.n)  # 2
    """

    # Check if 'problem_name' has the pattern '_{paramname}_{paramvalue}'. If it has, we load the problem with the specified SIF parameters.
    problem_name, params = _parse_problem_name(problem_name)
    if params:
        return pycutest_load(problem_name, **params)

    # Initialize CUTEst problem
    p = pycutest.import_problem(problem_name, destination=problem_name, sifParams=kwargs)

    fun = lambda x: p.obj(x)
    grad = lambda x: p.grad(x)
    hess = lambda x: p.ihess(x)

    # We replace 1.0e+20 (which represents infinity in pycutest) bounds with np.inf.
    p.bl = np.where(p.bl <= -1.0e+20, -np.inf, p.bl)
    p.bu = np.where(p.bu >= 1.0e+20, np.inf, p.bu)
    p.cl = np.where(p.cl <= -1.0e+20, -np.inf, p.cl) if p.m > 0 else p.cl
    p.cu = np.where(p.cu >= 1.0e+20, np.inf, p.cu) if p.m > 0 else p.cu

    x0 = p.x0
    xl = p.bl
    xu = p.bu

    cl = np.asarray(p.cl) if p.m > 0 else np.array([], dtype=float)
    cu = np.asarray(p.cu) if p.m > 0 else np.array([], dtype=float)
    mask_cl_finite = (cl > -np.inf) if p.m > 0 else np.array([], dtype=bool)
    mask_cu_finite = (cu < np.inf) if p.m > 0 else np.array([], dtype=bool)

    mask_linear = getattr(p, "is_linear_cons", None)
    if mask_linear is None:
        mask_linear = np.zeros(p.m, dtype=bool)
    mask_eq = getattr(p, "is_eq_cons", None)
    if mask_eq is None:
        mask_eq = np.zeros(p.m, dtype=bool)
    mask_linear_eq = mask_linear & mask_eq & mask_cl_finite
    mask_linear_ineq = mask_linear & ~mask_eq
    mask_linear_le = mask_linear_ineq & mask_cu_finite
    mask_linear_ge = mask_linear_ineq & mask_cl_finite
    mask_nonlinear_eq = ~mask_linear & mask_eq & mask_cl_finite
    mask_nonlinear_ineq = ~mask_linear & ~mask_eq
    mask_nonlinear_le = mask_nonlinear_ineq & mask_cu_finite
    mask_nonlinear_ge = mask_nonlinear_ineq & mask_cl_finite

    # The linear constraints are hidden in the cJx method output.
    # cx = jx @ x0 - bx
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            cx, jx = p.cons(x0, gradient=True)
            bx = jx @ x0 - cx
        except Exception:
            jx = None
            bx = None

    # Create the linear constraints if any.
    # Note that in PyCUTEst, the constraints are defined as:
    #  cl <= c(x) <= cu
    # Thus, the linear equality constraints are:
    #  jx[mask_linear_eq, :] @ x = bx[mask_linear_eq] + cu[mask_linear_eq]
    # and the linear inequality constraints are:
    #  jx[mask_linear_le, :] @ x <= bx[mask_linear_le] + cu[mask_linear_le]
    #  -jx[mask_linear_ge, :] @ x <= -bx[mask_linear_ge] - cl[mask_linear_ge]
    aeq = jx[mask_linear_eq, :] if np.any(mask_linear_eq) else np.zeros((0, p.n))
    beq = bx[mask_linear_eq] + cu[mask_linear_eq] if np.any(mask_linear_eq) else np.zeros(0)
    aub = np.vstack([jx[mask_linear_le, :], -jx[mask_linear_ge, :]]) if jx is not None and (np.any(mask_linear_le) or np.any(mask_linear_ge)) else np.zeros((0, p.n))
    bub = np.concatenate([bx[mask_linear_le] + cu[mask_linear_le], -bx[mask_linear_ge] - cl[mask_linear_ge]]) if bx is not None and (np.any(mask_linear_le) or np.any(mask_linear_ge)) else np.zeros(0)

    # Handle nonlinear constraints.
    # Construct nonlinear constraint functions
    # Remind that in S2MPJ, the constraints are defined as:
    #  cl <= c(x) <= cu
    # Thus, the nonlinear equality constraints are:
    #  ceq(x) = c(x)[mask_nonlinear_eq] - cu[mask_nonlinear_eq] = 0
    # and the nonlinear inequality constraints are:
    #  cub(x) = [c(x)[mask_nonlinear_le] - cu[mask_nonlinear_le];
    #           -c(x)[mask_nonlinear_ge] + cl[mask_nonlinear_ge]] <= 0
    def _process_nonlinear_ineq(x, mode="value"):
        """Helper for nonlinear inequality constraints
        mode in {"value", "jacobian", "hessian"}
        """
        if not np.any(mask_nonlinear_ineq):
            if mode == "value":
                return np.zeros(0)
            elif mode == "jacobian":
                return np.zeros((0, p.n))
            elif mode == "hessian":
                return []

        if mode == "value":
            c_all = p.cons(x)
            upper_vals = c_all[mask_nonlinear_le] - cu[mask_nonlinear_le]
            lower_vals = -c_all[mask_nonlinear_ge] + cl[mask_nonlinear_ge]
            return np.concatenate([upper_vals, lower_vals])
        elif mode == "jacobian":
            _, j_all = p.cons(x, gradient=True)
            j_upper = j_all[mask_nonlinear_le, :]
            j_lower = -j_all[mask_nonlinear_ge, :]
            return np.vstack([j_upper, j_lower]) if (j_upper.size > 0 or j_lower.size > 0) else np.zeros((0, p.n))
        elif mode == "hessian":
            hlist = []
            for i in np.where(mask_nonlinear_le)[0]:
                H = p.ihess(x, cons_index=i)
                hlist.append(H)
            for i in np.where(mask_nonlinear_ge)[0]:
                H = -p.ihess(x, cons_index=i)
                hlist.append(H)
            return hlist

    # Create the nonlinear equality constraints.
    def ceq(x):
        if not np.any(mask_nonlinear_eq):
            return np.zeros(0)
        c_all = p.cons(x)
        return c_all[mask_nonlinear_eq] - cu[mask_nonlinear_eq]
    
    # Create the nonlinear inequality constraints.
    def cub(x):
        return _process_nonlinear_ineq(x, "value")
    
    # Create Jacobian functions for nonlinear constraints.
    def jceq(x):
        if not np.any(mask_nonlinear_eq):
            return np.zeros((0, p.n))
        _, j_all = p.cons(x, gradient=True)
        return j_all[mask_nonlinear_eq, :]
    
    def jcub(x):
        return _process_nonlinear_ineq(x, "jacobian")
    
    # Create Hessian functions for nonlinear constraints.
    def hceq(x):
        if not np.any(mask_nonlinear_eq):
            return []
        idx_eq = np.where(mask_nonlinear_eq)[0]
        hlist = []
        for i in idx_eq:
            H = p.ihess(x, cons_index=i)
            hlist.append(H)
        return hlist

    def hcub(x):
        return _process_nonlinear_ineq(x, "hessian")

    problem = Problem(fun, x0, name=problem_name, xl=xl, xu=xu, aub=aub, bub=bub, aeq=aeq, beq=beq, cub=cub, ceq=ceq, grad=grad, hess=hess, jcub=jcub, jceq=jceq, hcub=hcub, hceq=hceq)

    return problem

def pycutest_select(options):
    """
    Select problems from PyCUTEst that satisfy given criteria.

    More details about PyCUTEst can be found at
    `the PyCUTEst documentation <https://jfowkes.github.io/pycutest/>`_.

    .. note::

        PyCUTEst is only available on Linux and macOS.

    Parameters
    ----------
    options : dict
        A dictionary containing selection criteria. Supported keys:

        - **ptype** (*str*) -- Type of problems to select. A string
          consisting of any combination of ``'u'`` (unconstrained),
          ``'b'`` (bound constrained), ``'l'`` (linearly constrained),
          and ``'n'`` (nonlinearly constrained), such as ``'b'``,
          ``'ul'``, ``'ubn'``. Default is ``'ubln'``.
        - **mindim** (*int*) -- Minimum dimension. Default is ``1``.
        - **maxdim** (*int*) -- Maximum dimension. Default is ``inf``.
        - **minb** (*int*) -- Minimum number of bound constraints.
          Default is ``0``.
        - **maxb** (*int*) -- Maximum number of bound constraints.
          Default is ``inf``.
        - **minlcon** (*int*) -- Minimum number of linear constraints.
          Default is ``0``.
        - **maxlcon** (*int*) -- Maximum number of linear constraints.
          Default is ``inf``.
        - **minnlcon** (*int*) -- Minimum number of nonlinear constraints.
          Default is ``0``.
        - **maxnlcon** (*int*) -- Maximum number of nonlinear constraints.
          Default is ``inf``.
        - **mincon** (*int*) -- Minimum total number of linear and
          nonlinear constraints. Default is ``0``.
        - **maxcon** (*int*) -- Maximum total number of linear and
          nonlinear constraints. Default is ``inf``.
        - **excludelist** (*list of str*) -- List of problem names to
          exclude. Default is ``[]``.

    Returns
    -------
    list of str
        Problem names that satisfy the given criteria.

    Notes
    -----
    1. PyCUTEst is only available on Linux and macOS.

    2. There is a file ``config.txt`` in the same directory as this
       module. It can be used to set the options ``variable_size`` and
       ``test_feasibility_problems``. See the comments in ``config.txt``
       for details. You can also override these options at runtime using
       `set_plib_config` or by setting environment variables
       ``PYCUTEST_VARIABLE_SIZE`` and
       ``PYCUTEST_TEST_FEASIBILITY_PROBLEMS``. Environment variables
       take precedence over ``config.txt``.

    See Also
    --------
    pycutest_load : Load a problem from PyCUTEst.
    optiprofiler.get_plib_config : Read the current configuration.
    optiprofiler.set_plib_config : Override configuration at runtime.

    Examples
    --------
    .. code-block:: python

        from optiprofiler.problem_libs.pycutest import pycutest_select

        names = pycutest_select({'ptype': 'u', 'maxdim': 5})
    """
    # Read config: environment variables (set via set_plib_config) take
    # precedence over the values in config.txt.
    variable_size = 'default'
    test_feasibility_problems = 0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.txt')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('variable_size='):
                        variable_size = stripped.split('=')[1].split('#')[0].split('%')[0].strip()
                    elif stripped.startswith('test_feasibility_problems='):
                        test_feasibility_problems = int(stripped.split('=')[1].split('#')[0].split('%')[0].strip())
        except Exception:
            pass
    if 'PYCUTEST_VARIABLE_SIZE' in os.environ:
        variable_size = os.environ['PYCUTEST_VARIABLE_SIZE']
    if 'PYCUTEST_TEST_FEASIBILITY_PROBLEMS' in os.environ:
        test_feasibility_problems = int(os.environ['PYCUTEST_TEST_FEASIBILITY_PROBLEMS'])
    
    if variable_size not in ['default', 'min', 'max', 'all']:
        raise ValueError("Invalid `variable_size` in the file `config.txt`. Please set it to 'default', 'min', 'max', or 'all'.")
    if test_feasibility_problems not in [0, 1, 2]:
        raise ValueError("Invalid `test_feasibility_problems` in the file `config.txt`. Please set it to 0, 1, or 2.")
    
    # Initialize result lists
    problem_names = []
    
    # Set default values for options
    default_options = {
        'ptype': 'ubln',
        'mindim': 1,
        'maxdim': np.inf,
        'minb': 0,
        'maxb': np.inf,
        'minlcon': 0,
        'maxlcon': np.inf,
        'minnlcon': 0, 
        'maxnlcon': np.inf,
        'mincon': 0,
        'maxcon': np.inf,
        'excludelist': []
    }
    for key in default_options:
        options.setdefault(key, default_options[key])
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load problem info from CSV file
    try:
        probinfo_path = os.path.join(current_dir, 'probinfo_pycutest.csv')
        probinfo = pd.read_csv(probinfo_path)
    except:
        raise FileNotFoundError(f"Could not find or load problem info file at {probinfo_path}")
    
    # Helper function to safely convert values
    def safe_convert(value, default=0):
        if pd.isna(value) or value == 'unknown':
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

    # Process each problem
    for _, row in probinfo.iterrows():
        # Get problem default attributes
        problem_name = row['problem_name']
        ptype = row['ptype']
        is_feasibility = row['isfeasibility']
        dim = safe_convert(row['dim'])
        mb = safe_convert(row['mb'])
        mlcon = safe_convert(row['mlcon'])
        mnlcon = safe_convert(row['mnlcon'])
        mcon = safe_convert(row['mcon'])

        # Check if argins (variable sizes) exist. If exist, convert them to numeric
        # `row['argins']` is in the format: '{'N':10,'B':5}{'N':10,'B':10} ...'
        argins_str = row['argins'] if pd.notna(row['argins']) else ''
        argins = []
        # We convert argins_str to a list of strings in the format: ['N_10_B_5', 'N_10_B_10', ...]
        if argins_str:
            argins_parts = re.findall(r'\{([^}]+)\}', argins_str)
            for part in argins_parts:
                cleaned_str = part.replace("'", "").replace('"', '').replace(' ', '')
                cleaned_str = cleaned_str.replace(':', '_').replace(',', '_')
                argins.append(cleaned_str)

        dims = row['dims'].split() if pd.notna(row['dims']) else []
        mbs = row['mbs'].split() if pd.notna(row['mbs']) else []
        mlcons = row['mlcons'].split() if pd.notna(row['mlcons']) else []
        mnlcons = row['mnlcons'].split() if pd.notna(row['mnlcons']) else []
        mcons = row['mcons'].split() if pd.notna(row['mcons']) else []
        if dims:
            dims = [safe_convert(d) for d in dims]
        if mbs:
            mbs = [safe_convert(m) for m in mbs]
        if mlcons:
            mlcons = [safe_convert(m) for m in mlcons]
        if mnlcons:
            mnlcons = [safe_convert(m) for m in mnlcons]
        if mcons:
            mcons = [safe_convert(m) for m in mcons]

        # Check problem type
        if ptype not in options['ptype']:
            continue

        # Check feasibility problems based on user preference
        if test_feasibility_problems == 0:
            if is_feasibility:
                continue
        elif test_feasibility_problems == 1:
            if not is_feasibility:
                continue
        # If test_feasibility_problems == 2, do nothing (include all problems)

        # Check if default dimension and constraints satisfy criteria
        default_satisfy = (
            dim >= options['mindim'] and dim <= options['maxdim'] and
            mb >= options['minb'] and mb <= options['maxb'] and
            mlcon >= options['minlcon'] and mlcon <= options['maxlcon'] and
            mnlcon >= options['minnlcon'] and mnlcon <= options['maxnlcon'] and
            mcon >= options['mincon'] and mcon <= options['maxcon'] and
            problem_name not in options['excludelist']
        )
        
        # If default satisfies and (no variable sizes or we only want default), add the problem
        if default_satisfy and (not dims or variable_size == 'default'):
            problem_names.append(problem_name)
        
        # Skip variable size processing if we only want default
        if variable_size == 'default':
            continue
        
        # Process variable sizes if they exist
        if dims:
            # Create mask for configurations that satisfy criteria
            mask = []
            names = [f"{problem_name}_{arg}" for arg in argins] if argins else [problem_name]*len(dims)

            for i in range(len(dims)):
                # Create the name for the current configuration
                name = names[i] if i < len(names) else problem_name
                satisfies = (
                    dims[i] >= options['mindim'] and dims[i] <= options['maxdim'] and
                    (i >= len(mbs) or mbs[i] >= options['minb']) and
                    (i >= len(mbs) or mbs[i] <= options['maxb']) and
                    (i >= len(mlcons) or mlcons[i] >= options['minlcon']) and
                    (i >= len(mlcons) or mlcons[i] <= options['maxlcon']) and
                    (i >= len(mnlcons) or mnlcons[i] >= options['minnlcon']) and
                    (i >= len(mnlcons) or mnlcons[i] <= options['maxnlcon']) and
                    (i >= len(mcons) or mcons[i] >= options['mincon']) and
                    (i >= len(mcons) or mcons[i] <= options['maxcon']) and
                    name not in options['excludelist']
                )
                mask.append(satisfies)
            
            idxs = [i for i, m in enumerate(mask) if m]
            
            if not idxs:
                if default_satisfy:
                    problem_names.append(problem_name)
                continue
            
            if variable_size == 'min':
                # Find minimum dimension among configurations that satisfy
                min_dim = min(dims[i] for i in idxs)
                idxs_dim = [i for i in idxs if dims[i] == min_dim]
                
                # Find minimum constraints among those
                if mcons and idxs_dim:
                    min_mcon = min(mcons[i] if i < len(mcons) else 0 for i in idxs_dim)
                    idxs = [i for i in idxs_dim if i >= len(mcons) or mcons[i] == min_mcon]
                    idxs = [idxs[0]]  # Take just the first
                    
                    # Compare with default
                    if default_satisfy and (dim < min_dim or (dim == min_dim and mcon < min_mcon)):
                        problem_names.append(problem_name)
                        continue
            
            elif variable_size == 'max':
                # Find maximum dimension among configurations that satisfy
                max_dim = max(dims[i] for i in idxs)
                idxs_dim = [i for i in idxs if dims[i] == max_dim]
                
                # Find maximum constraints among those
                if mcons and idxs_dim:
                    max_mcon = max(mcons[i] if i < len(mcons) else 0 for i in idxs_dim)
                    idxs = [i for i in idxs_dim if i < len(mcons) and mcons[i] == max_mcon]
                    idxs = [idxs[0]]  # Take just the first
                    
                    # Compare with default
                    if default_satisfy and (dim > max_dim or (dim == max_dim and mcon > max_mcon)):
                        problem_names.append(problem_name)
                        continue
            
            # For 'all' we just process all idxs
            
            # Add problems with specific dimensions/constraints
            for idx in idxs:
                if argins and idx < len(argins):
                    problem_names.append(names[idx])
    
    return problem_names

# Define a function to collect SIF parameters
def pycutest_get_sif_params(problem_name):
    """
    Parse the printed output of pycutest.print_available_sif_params(problem_name)
    and extract available SIF parameters.

    Parameters
    ----------
    problem_name : str
        The name of the problem in pycutest.

    Returns
    -------
    para_names : list of str
        list of parameter names (e.g., ['A', 'N'])
    para_values : list of list
        list of possible values, each is a list (e.g., [[1, 2, 3], [10, 20, 40]])
    para_defaults : list
        list of default values (e.g., [2, 20])
    If the problem has no SIF parameters, all three lists are empty.
    """

    problem_name, params = _parse_problem_name(problem_name)
    if params:
        return pycutest_get_sif_params(problem_name)

    # Capture the printed output of pycutest.print_available_sif_params
    buf = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buf
    try:
        pycutest.print_available_sif_params(problem_name)
    finally:
        sys.stdout = sys_stdout

    lines = buf.getvalue().splitlines()
    buf.close()

    # Updated pattern — now we capture also the optional type (inside parentheses)
    # Example line: "N = 40 (int) [default]"
    # Exclude lines where parentheses contain 'for' (e.g., "(for SIF parameter N=1)")
    pattern = re.compile(
        r"^\s*([^\s=]+)\s*=\s*([^\s\(]+)"   # param name and value
        r"(?:\s*\((?!.*\bfor\b)([^)]*)\))?" # optional '(int)' or '(float)' group, negative lookahead to avoid 'for'
        r"\s*(\[default\])?\s*$"            # optional [default]
    )

    params = {}    # dict: param_name -> list of values
    defaults = {}  # dict: param_name -> default value

    for line in lines:
        m = pattern.search(line)
        if not m:
            continue

        key = m.group(1)                # parameter name
        val_raw = m.group(2)            # numeric/string value
        type_hint = m.group(3) or ""    # 'int', 'float', maybe empty
        default_flag = bool(m.group(4)) # whether [default] exists

        # Determine type conversion priority:
        # 1. explicit type hint in parentheses (best)
        # 2. fallback heuristic based on the string itself
        val = val_raw
        try:
            if "float" in type_hint.lower():
                val = float(val_raw)
            elif "int" in type_hint.lower():
                val = int(val_raw)
            else:
                # fallback heuristic if no type specified
                if "." in val_raw or "e" in val_raw.lower():
                    val = float(val_raw)
                else:
                    val = int(val_raw)
        except ValueError:
            # fallback to original string if conversion fails
            val = val_raw

        params.setdefault(key, []).append(val)
        if default_flag:
            defaults[key] = val

    # Return empty lists if no parameters found
    if not params:
        return [], [], []
    
    # Remove duplicate values while preserving order
    for key in params:
        seen = set()
        unique_values = []
        for val in params[key]:
            if val not in seen:
                seen.add(val)
                unique_values.append(val)
        params[key] = unique_values

    # Keep order of appearance
    para_names = list(params.keys())
    para_values = [params[k] for k in para_names]
    para_defaults = [defaults.get(k) for k in para_names]

    # Filter out parameters with only one value
    filtered_names = []
    filtered_values = []
    filtered_defaults = []
    
    for name, values, default in zip(para_names, para_values, para_defaults):
        if len(values) > 1:
            filtered_names.append(name)
            filtered_values.append(values)
            filtered_defaults.append(default)

    return filtered_names, filtered_values, filtered_defaults

def pycutest_clear_cache(problem_name, **kwargs):
    problem_name, params = _parse_problem_name(problem_name)
    if params:
        pycutest_clear_cache(problem_name, **params)
    else:
        pycutest.clear_cache(problem_name, sifParams=kwargs)

def pycutest_clear_all_cache():
    # Delete the folder 'pycutest_cache_holder' directly.
    cache_holder_path = os.path.join(cache_dir, 'pycutest_cache_holder')
    if os.path.exists(cache_holder_path):
        shutil.rmtree(cache_holder_path)

def _parse_problem_name(problem_name):
    # Check if 'problem_name' has the pattern '_{paramname}_{paramvalue}'. If it has, we separate the problem name and its sif parameters (and their values).
    base_name = problem_name
    params = {}

    if '_' in problem_name:
        parts = problem_name.split('_')
        base_name = parts[0]

        if len(parts) > 1 and len(parts[1:]) % 2 == 0:
            
            for i_param in range(1, len(parts), 2):
                param_name = parts[i_param]
                param_value = parts[i_param + 1]
                # Try to convert to int or float if possible
                try:
                    value_float = float(param_value)
                    if value_float.is_integer():
                        param_value = int(value_float)
                    else:
                        param_value = value_float
                except ValueError:
                    raise ValueError(f"Invalid problem name, which contains non-numeric SIF parameter value: {problem_name}")
                params[param_name] = param_value
            
    return base_name, params