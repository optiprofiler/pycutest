"""
p_getInfo.py - Collect problem information from PyCUTEst problem set

This script scans all problems in the PyCUTEst collection and extracts
various metrics including dimensions, constraint counts, and function values.
The results are saved to CSV files for later use by OptiProfiler.

Key features for robustness:
- Block-based processing to handle large number of problems
- Timeout protection for each problem
- Memory cleanup after each problem
- Blacklist for known problematic problems
- Graceful error handling

Usage: python p_getInfo.py
"""

import io
import re
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import TimeoutError
import threading
import gc
import shutil

# Get the repository root directory (three levels up from this script)
cwd = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(cwd, '..', '..', '..'))

# Set CUTEst environment variables BEFORE importing pycutest
# This is critical because pycutest checks these variables at import time
# Use $HOME if available, otherwise fall back to ~ expansion
home_dir = os.environ.get('HOME', os.path.expanduser('~'))
cutest_base = os.path.join(home_dir, 'cutest')

if 'ARCHDEFS' not in os.environ:
    os.environ['ARCHDEFS'] = os.path.join(cutest_base, 'archdefs')
if 'SIFDECODE' not in os.environ:
    os.environ['SIFDECODE'] = os.path.join(cutest_base, 'sifdecode')
if 'CUTEST' not in os.environ:
    os.environ['CUTEST'] = os.path.join(cutest_base, 'cutest')
if 'MASTSIF' not in os.environ:
    os.environ['MASTSIF'] = os.path.join(cutest_base, 'mastsif', 'sif')
if 'MYARCH' not in os.environ:
    os.environ['MYARCH'] = 'pc64.lnx.gfo'

# Add CUTEst binaries to PATH if not already there
cutest_bin_paths = [
    os.path.join(cutest_base, 'sifdecode', 'bin'),
    os.path.join(cutest_base, 'cutest', 'bin')
]
current_path = os.environ.get('PATH', '')
for bin_path in cutest_bin_paths:
    if bin_path not in current_path:
        os.environ['PATH'] = f"{bin_path}:{current_path}" if current_path else bin_path

# Add optiprofiler to the system path (checked out by GitHub Actions)
sys.path.append(os.path.join(repo_root, 'optiprofiler'))
sys.path.append(os.path.join(repo_root, 'optiprofiler', 'python'))

import optiprofiler
from optiprofiler.problem_libs.pycutest.pycutest_tools import (
    pycutest_load, pycutest_select, pycutest_get_sif_params,
    pycutest_clear_cache, pycutest_clear_all_cache
)

# ============================================================================
# Configuration
# ============================================================================

# Number of blocks to split problems into (for memory management)
num_blocks = 20

# Timeout (seconds) for each problem to be loaded
timeout = 30

# Output path (repository root)
saving_path = repo_root

# Problems known to cause timeout or memory issues - skip these entirely
timeout_memory_blacklist = [
    'BA-L52', 'BA-L16', 'BA-L16LS', 'BA-L52LS', 'BDRY2', 'NET4', 'PDE1', 'PDE2'
]

# List of known feasibility problems (objective function is not meaningful)
known_feasibility = [
    'AIRCRFTA', 'ARGAUSS', 'ARGLALE', 'ARGLBLE', 'ARGTRIG', 'ARTIF', 'BAmL1SP',
    'BARDNE', 'BEALENE', 'BENNETT5', 'BIGGS6NE', 'BOOTH', 'BOXBOD', 'BRATU2D',
    'BRATU2DT', 'BRATU3D', 'BROWNBSNE', 'BROWNDENE', 'BROYDN3D', 'CBRATU2D',
    'CBRATU3D', 'CHANDHEQ', 'CHEMRCTA', 'CHWIRUT2', 'CLUSTER', 'COOLHANS',
    'CUBENE', 'CYCLIC3', 'CYCLOOCF', 'CYCLOOCT', 'DANIWOOD', 'DANWOOD',
    'DECONVBNE', 'DENSCHNBNE', 'DENSCHNDNE', 'DENSCHNFNE', 'DEVGLA1NE',
    'DEVGLA2NE', 'DRCAVTY1', 'DRCAVTY2', 'DRCAVTY3', 'ECKERLE4', 'EGGCRATENE',
    'EIGENA', 'EIGENB', 'ELATVIDUNE', 'ENGVAL2NE', 'ENSO', 'ERRINROSNE',
    'ERRINRSMNE', 'EXP2NE', 'EXTROSNBNE', 'FLOSP2HH', 'FLOSP2HL', 'FLOSP2HM',
    'FLOSP2TH', 'FLOSP2TL', 'FLOSP2TM', 'FREURONE', 'GENROSEBNE', 'GOTTFR',
    'GROWTH', 'GULFNE', 'HAHN1', 'HATFLDANE', 'HATFLDBNE', 'HATFLDCNE',
    'HATFLDDNE', 'HATFLDENE', 'HATFLDFLNE', 'HATFLDF', 'HATFLDG', 'HELIXNE',
    'HIMMELBA', 'HIMMELBC', 'HIMMELBD', 'HIMMELBFNE', 'HS1NE', 'HS25NE',
    'HS2NE', 'HS8', 'HYDCAR20', 'HYDCAR6', 'HYPCIR', 'INTEGREQ', 'INTEQNE',
    'KOEBHELBNE', 'KOWOSBNE', 'KSS', 'LANCZOS1', 'LANCZOS2', 'LANCZOS3',
    'LEVYMONE10', 'LEVYMONE5', 'LEVYMONE6', 'LEVYMONE7', 'LEVYMONE8',
    'LEVYMONE9', 'LEVYMONE', 'LIARWHDNE', 'LINVERSENE', 'LSC1', 'LSC2',
    'LUKSAN11', 'LUKSAN12', 'LUKSAN13', 'LUKSAN14', 'LUKSAN17', 'LUKSAN21',
    'LUKSAN22', 'MANCINONE', 'METHANB8', 'METHANL8', 'MEYER3NE', 'MGH09',
    'MGH10', 'MISRA1A', 'MISRA1B', 'MISRA1C', 'MISRA1D', 'MODBEALENE',
    'MSQRTA', 'MSQRTB', 'MUONSINE', 'n10FOLDTR', 'NELSON', 'NONSCOMPNE',
    'NYSTROM5', 'OSBORNE1', 'OSBORNE2', 'OSCIGRNE', 'OSCIPANE', 'PALMER1ANE',
    'PALMER1BNE', 'PALMER1ENE', 'PALMER1NE', 'PALMER2ANE', 'PALMER2BNE',
    'PALMER2ENE', 'PALMER3ANE', 'PALMER3BNE', 'PALMER3ENE', 'PALMER4ANE',
    'PALMER4BNE', 'PALMER4ENE', 'PALMER5ANE', 'PALMER5BNE', 'PALMER5ENE',
    'PALMER6ANE', 'PALMER6ENE', 'PALMER7ANE', 'PALMER7ENE', 'PALMER8ANE',
    'PALMER8ENE', 'PENLT1NE', 'PENLT2NE', 'POROUS1', 'POROUS2', 'POWELLBS',
    'POWELLSQ', 'POWERSUMNE', 'PRICE3NE', 'PRICE4NE', 'QINGNE', 'QR3D',
    'RAT42', 'RAT43', 'RECIPE', 'REPEAT', 'RES', 'ROSZMAN1', 'RSNBRNE',
    'SANTA', 'SEMICN2U', 'SEMICON1', 'SEMICON2', 'SPECANNE', 'SSBRYBNDNE',
    'SSINE', 'THURBER', 'TQUARTICNE', 'VANDERM1', 'VANDERM2', 'VANDERM3',
    'VANDERM4', 'VARDIMNE', 'VESUVIA', 'VESUVIO', 'VESUVIOU', 'VIBRBEAMNE',
    'WATSONNE', 'WAYSEA1NE', 'WAYSEA2NE', 'YATP1CNE', 'YATP2CNE', 'YFITNE',
    'ZANGWIL3'
]

# Global lists for tracking
feasibility = []
timeout_problems = []


# ============================================================================
# Logging
# ============================================================================

class Logger:
    """Dual-output logger that writes to both terminal and log file."""
    def __init__(self, logfile):
        self.terminal = sys.__stdout__
        self.log = logfile

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message)
        except Exception as e:
            self.terminal.write(f"[Logger Error] {e}\n")

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ============================================================================
# Timeout handling
# ============================================================================

def run_with_timeout(func, args, timeout_seconds):
    """
    Execute a function with a timeout using threading.
    
    This is more robust than signal-based timeout as it works
    in multi-threaded environments.
    """
    result = [None]
    exception = [None]

    def wrapper():
        try:
            result[0] = func(*args) if args else func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(f"Function with args {args} timed out after {timeout_seconds} seconds.")
        raise TimeoutError(f"Timeout after {timeout_seconds} seconds")

    if exception[0] is not None:
        raise exception[0]

    return result[0]


# ============================================================================
# Problem parameter filtering (to avoid memory issues)
# ============================================================================

def should_skip_parameter_combination(problem_name, comb):
    """
    Check if a parameter combination should be skipped to avoid known issues.
    
    Returns True if the combination should be skipped.
    """
    # Skip if any parameter value is too large
    for val in comb:
        try:
            if np.isscalar(val) and abs(val) >= 1e5:
                return True
        except Exception:
            continue

    # Problem-specific skips based on known memory/timeout issues
    skip_rules = {
        'ALLINQP': lambda c: c[0] >= 100000,
        'AUG2D': lambda c: len(c) >= 2 and c[0] >= 200 and c[1] >= 200,
        'AUG2DC': lambda c: len(c) >= 2 and c[0] >= 200 and c[1] >= 200,
        'AUG2DCQP': lambda c: len(c) >= 2 and c[0] >= 200 and c[1] >= 200,
        'AUG2DQP': lambda c: len(c) >= 2 and c[0] >= 200 and c[1] >= 200,
        'AUG3DC': lambda c: len(c) >= 3 and c[0] >= 30 and c[1] >= 20 and c[2] >= 30,
        'AUG3DQP': lambda c: len(c) >= 3 and c[0] >= 30 and c[1] >= 20 and c[2] >= 30,
        'AUG3DCQP': lambda c: len(c) >= 3 and c[0] >= 20 and c[1] >= 20 and c[2] >= 20,
        'BLOCKQP4': lambda c: c[0] >= 10000,
        'BLOCKQP5': lambda c: c[0] >= 10000,
        'BDVALUES': lambda c: len(c) >= 2 and c[0] >= 1e4 and c[1] >= 100,
        'CHARDIS0': lambda c: c[0] >= 2000,
        'CHARDIS1': lambda c: c[0] >= 2000,
        'CONT5-QP': lambda c: c[0] >= 400,
        'CONT6-QQ': lambda c: c[0] >= 400,
        'DTOC1NC': lambda c: len(c) >= 3 and c[0] >= 1000 and c[1] >= 5 and c[2] >= 10,
        'GAUSSELM': lambda c: c[0] >= 50,
        'HARKERP2': lambda c: c[0] >= 5000,
        'JUNKTURN': lambda c: c[0] >= 100000,
        'LUKVLE13': lambda c: c[0] >= 99998,
        'LUKVLI14': lambda c: c[0] >= 99998,
        'LUKVLI15': lambda c: c[0] >= 99997,
        'LUKVLI16': lambda c: c[0] >= 99997,
        'LUKVLI17': lambda c: c[0] >= 99997,
        'NUFFIELD': lambda c: c[0] >= 100,
        'OPTCTRL6': lambda c: c[0] >= 50000,
        'ORTHREGA': lambda c: c[0] >= 8,
        'ORTHREGC': lambda c: c[0] >= 50000,
        'RDW2D51F': lambda c: c[0] >= 512,
        'RDW2D51U': lambda c: c[0] >= 512,
        'RDW2D52B': lambda c: c[0] >= 512,
        'RDW2D52F': lambda c: c[0] >= 512,
        'RDW2D52U': lambda c: c[0] >= 512,
        'ROSEPETAL': lambda c: c[0] >= 10000,
        'TWOD': lambda c: c[0] >= 79,
        'SENSORS': lambda c: c[0] >= 1000,
        'SOSQP1': lambda c: c[0] >= 50000,
        'SOSQP2': lambda c: c[0] >= 50000,
        'STCQP2': lambda c: c[0] >= 16,
        'STNQP1': lambda c: c[0] >= 16,
    }

    if problem_name in skip_rules:
        try:
            return skip_rules[problem_name](comb)
        except (IndexError, TypeError):
            return False

    return False


# ============================================================================
# Problem information extraction
# ============================================================================

def get_problem_info(problem_name, para_names=None, para_values=None, para_defaults=None):
    """
    Extract information about a single problem.
    
    This function handles both non-parametric and parametric problems,
    with proper timeout and memory management.
    """
    global feasibility, timeout_problems

    print(f"Processing problem: {problem_name}")

    # Initialize info dictionary with default values
    info_single = {
        'problem_name': problem_name,
        'ptype': 'unknown',
        'xtype': 'unknown',
        'dim': 'unknown',
        'mb': 'unknown',
        'ml': 'unknown',
        'mu': 'unknown',
        'mcon': 'unknown',
        'mlcon': 'unknown',
        'mnlcon': 'unknown',
        'm_ub': 'unknown',
        'm_eq': 'unknown',
        'm_linear_ub': 'unknown',
        'm_linear_eq': 'unknown',
        'm_nonlinear_ub': 'unknown',
        'm_nonlinear_eq': 'unknown',
        'f0': 0,
        'isfeasibility': 1,
        'isgrad': 1,
        'ishess': 1,
        'isjcub': 1,
        'isjceq': 1,
        'ishcub': 1,
        'ishceq': 1,
        'argins': '',
        'dims': '',
        'mbs': '',
        'mls': '',
        'mus': '',
        'mcons': '',
        'mlcons': '',
        'mnlcons': '',
        'm_ubs': '',
        'm_eqs': '',
        'm_linear_ubs': '',
        'm_linear_eqs': '',
        'm_nonlinear_ubs': '',
        'm_nonlinear_eqs': '',
        'f0s': ''
    }

    # Try to load the default problem
    p = None
    try:
        p = run_with_timeout(pycutest_load, (problem_name,), timeout)
    except TimeoutError:
        print(f"Timeout while loading problem {problem_name}.")
        timeout_problems.append(problem_name)
        return info_single
    except Exception as e:
        print(f"Error loading problem {problem_name}: {e}")
        return info_single

    # Extract basic problem information
    try:
        info_single['ptype'] = p.ptype
        info_single['xtype'] = 'r'
        info_single['dim'] = p.n
        info_single['mb'] = p.mb
        info_single['ml'] = sum(p.xl > -np.inf)
        info_single['mu'] = sum(p.xu < np.inf)
        info_single['mcon'] = p.mcon
        info_single['mlcon'] = p.mlcon
        info_single['mnlcon'] = p.mnlcon
        info_single['m_ub'] = p.m_linear_ub + p.m_nonlinear_ub
        info_single['m_eq'] = p.m_linear_eq + p.m_nonlinear_eq
        info_single['m_linear_ub'] = p.m_linear_ub
        info_single['m_linear_eq'] = p.m_linear_eq
        info_single['m_nonlinear_ub'] = p.m_nonlinear_ub
        info_single['m_nonlinear_eq'] = p.m_nonlinear_eq
    except Exception as e:
        print(f"Error getting problem info for {problem_name}: {e}")

    # Evaluate objective function to check feasibility
    try:
        f = p.fun(p.x0)
        if problem_name == 'LIN':
            info_single['isfeasibility'] = 0
            info_single['f0'] = np.nan
        elif np.size(f) == 0 or np.isnan(f) or problem_name in known_feasibility:
            info_single['isfeasibility'] = 1
            info_single['f0'] = 0
            feasibility.append(problem_name)
        else:
            info_single['isfeasibility'] = 0
            info_single['f0'] = f
    except Exception as e:
        print(f"Error evaluating function for {problem_name}: {e}")
        info_single['f0'] = 0
        info_single['isfeasibility'] = 1
        feasibility.append(problem_name)

    # Clean up default problem
    try:
        pycutest_clear_cache(problem_name)
    except:
        pass
    del p
    gc.collect()

    # Handle parametric problems if parameters exist
    if para_names is None or len(para_names) == 0:
        print(f"Finished processing problem {problem_name} (non-parametric).")
        return info_single

    print(f"Processing parametric problem: {problem_name} with parameters: {para_names}")

    # Generate parameter combinations
    para_combinations = np.array(np.meshgrid(*para_values)).T.reshape(-1, len(para_names))
    
    # Filter out default combinations
    nondefault_para_combinations = []
    if para_defaults is not None and all(d is not None for d in para_defaults):
        for comb in para_combinations:
            if not all(comb[i] == para_defaults[i] for i in range(len(para_names))):
                nondefault_para_combinations.append(comb)
    else:
        nondefault_para_combinations = list(para_combinations)

    successful_para_combinations = []

    for comb in nondefault_para_combinations:
        # Check if this combination should be skipped
        if should_skip_parameter_combination(problem_name, comb):
            print(f"Skipping parameter combination {comb} for {problem_name} (known issues)")
            continue

        print(f"Processing {problem_name} with params: {dict(zip(para_names, comb))}")

        try:
            result = process_parametric_problem(problem_name, para_names, comb)
            if result is None:
                continue

            successful_para_combinations.append(comb)
            info_single['dims'] += str(result['n']) + ' '
            info_single['mbs'] += str(result['mb']) + ' '
            info_single['mls'] += str(result['ml']) + ' '
            info_single['mus'] += str(result['mu']) + ' '
            info_single['mcons'] += str(result['mcon']) + ' '
            info_single['mlcons'] += str(result['mlcon']) + ' '
            info_single['mnlcons'] += str(result['mnlcon']) + ' '
            info_single['m_ubs'] += str(result['m_ub']) + ' '
            info_single['m_eqs'] += str(result['m_eq']) + ' '
            info_single['m_linear_ubs'] += str(result['m_linear_ub']) + ' '
            info_single['m_linear_eqs'] += str(result['m_linear_eq']) + ' '
            info_single['m_nonlinear_ubs'] += str(result['m_nonlinear_ub']) + ' '
            info_single['m_nonlinear_eqs'] += str(result['m_nonlinear_eq']) + ' '
            info_single['f0s'] += str(result['f0']) + ' '

        except TimeoutError:
            print(f"Timeout for {problem_name} with params {comb}")
            timeout_problems.append(f"{problem_name} with params {comb}")
        except Exception as e:
            print(f"Error processing {problem_name} with params {comb}: {e}")
        finally:
            # Always clean up
            try:
                pycutest_clear_cache(problem_name, **dict(zip(para_names, comb)))
            except:
                pass
            gc.collect()

    # Format argins string
    arg_strs = ''
    for comb in successful_para_combinations:
        arg_str = '{' + ','.join([f"'{para_names[i]}':{comb[i]}" for i in range(len(para_names))]) + '}'
        arg_strs += arg_str
    info_single['argins'] = arg_strs.strip()

    # Strip trailing spaces from all multi-value fields
    for key in ['dims', 'mbs', 'mls', 'mus', 'mcons', 'mlcons', 'mnlcons',
                'm_ubs', 'm_eqs', 'm_linear_ubs', 'm_linear_eqs',
                'm_nonlinear_ubs', 'm_nonlinear_eqs', 'f0s']:
        info_single[key] = info_single[key].strip()

    print(f"Finished processing problem {problem_name} (parametric).")
    return info_single


def process_parametric_problem(problem_name, para_names, comb):
    """
    Process a single parametric problem configuration.
    
    Returns a dict with problem info, or None on failure.
    """
    def _load_and_extract():
        para_dict = dict(zip(para_names, comb))
        p = pycutest_load(problem_name, **para_dict)

        result = {}
        result['n'] = p.n
        result['mb'] = p.mb
        result['ml'] = sum(p.xl > -np.inf)
        result['mu'] = sum(p.xu < np.inf)

        try:
            result['mcon'] = p.mcon
        except AttributeError:
            result['mcon'] = p.mlcon + p.m_nonlinear_ub + p.m_nonlinear_eq

        result['mlcon'] = p.mlcon

        try:
            result['mnlcon'] = p.mnlcon
        except AttributeError:
            result['mnlcon'] = p.m_nonlinear_ub + p.m_nonlinear_eq

        result['m_ub'] = p.m_linear_ub + p.m_nonlinear_ub
        result['m_eq'] = p.m_linear_eq + p.m_nonlinear_eq
        result['m_linear_ub'] = p.m_linear_ub
        result['m_linear_eq'] = p.m_linear_eq
        result['m_nonlinear_ub'] = p.m_nonlinear_ub
        result['m_nonlinear_eq'] = p.m_nonlinear_eq

        if problem_name in known_feasibility:
            result['f0'] = 0
        else:
            f = p.fun(p.x0)
            result['f0'] = 0 if (np.size(f) == 0 or np.isnan(f)) else f

        return result

    return run_with_timeout(_load_and_extract, (), timeout)


# ============================================================================
# Block-based processing
# ============================================================================

def collect_problem_info(block_number, problem_names_all):
    """
    Collect problem info for a specific block of problems.
    
    This divides the total problem set into blocks to manage memory.
    """
    total_problems = len(problem_names_all)
    block_size = (total_problems + num_blocks - 1) // num_blocks
    start_idx = block_number * block_size
    end_idx = min(start_idx + block_size, total_problems)
    problem_names = problem_names_all[start_idx:end_idx]

    print(f"Block {block_number}: Processing problems {start_idx} to {end_idx - 1}")
    print(f"Problems: {problem_names}")

    results = []
    for name in problem_names:
        print(f"\n>>> STARTING: {name}")
        sys.stdout.flush()

        para_names, para_values, para_defaults = pycutest_get_sif_params(name)
        info = get_problem_info(name, para_names, para_values, para_defaults)
        results.append(info)

        sys.stdout.flush()
        sys.stderr.flush()

    # Save block results
    df = pd.DataFrame(results)

    def has_unknown_values(row):
        for value in row:
            if str(value).strip().lower() == 'unknown':
                return True
        return False

    unknown_mask = df.apply(has_unknown_values, axis=1)
    if unknown_mask.any():
        filtered_problems = df.loc[unknown_mask, 'problem_name'].tolist()
        print(f"Filtered out {len(filtered_problems)} problems with 'unknown' values:")
        for problem in filtered_problems:
            print(f"  - {problem}")

    df_clean = df[~unknown_mask]
    df_clean.to_csv(os.path.join(saving_path, f'probinfo_pycutest_block{block_number}.csv'),
                    index=False, na_rep='nan')

    # Save auxiliary files
    with open(os.path.join(saving_path, 'feasibility_pycutest.txt'), 'w') as f:
        f.write(' '.join(feasibility))

    with open(os.path.join(saving_path, 'timeout_problems_pycutest.txt'), 'w') as f:
        f.write(' '.join(timeout_problems))

    print(f"Block {block_number} completed successfully.")


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Collect problem info from PyCUTEst.')
    parser.add_argument('--block', type=int, default=None, help='Block number to process (0 to num_blocks-1)')
    parser.add_argument('--merge', action='store_true', help='Merge all block files into one CSV')
    args_cmd = parser.parse_args()

    # Set up logging
    log_name = 'log_pycutest.txt' if args_cmd.block is None else f'log_pycutest_block{args_cmd.block}.txt'
    log_file = open(os.path.join(saving_path, log_name), 'w')
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)

    # Get all problem names (excluding blacklisted ones)
    problem_names_all = pycutest_select({})
    problem_names_all = [name for name in problem_names_all if name not in timeout_memory_blacklist]
    problem_names_all.sort()

    print(f"Total problems to process: {len(problem_names_all)}")

    if args_cmd.merge:
        # Merge all block CSV files
        print("\n=== Merging all block files ===")
        dfs = []
        for i in range(num_blocks):
            block_file = os.path.join(saving_path, f'probinfo_pycutest_block{i}.csv')
            if os.path.exists(block_file):
                print(f"Adding {block_file}")
                dfs.append(pd.read_csv(block_file))
            else:
                print(f"Warning: {block_file} not found")

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            # Remove duplicates just in case
            merged_df = merged_df.drop_duplicates(subset=['problem_name'])
            merged_df.sort_values(by='problem_name', inplace=True)
            merged_df.to_csv(os.path.join(saving_path, 'probinfo_pycutest.csv'), index=False)
            print(f"Merged {len(dfs)} block files into probinfo_pycutest.csv")

            # Merge feasibility and timeout files
            all_feasibility = []
            all_timeout = []
            for i in range(num_blocks):
                feas_file = os.path.join(saving_path, f'feasibility_pycutest_block{i}.txt')
                if os.path.exists(feas_file):
                    with open(feas_file, 'r') as f:
                        all_feasibility.extend(f.read().split())
                
                tout_file = os.path.join(saving_path, f'timeout_problems_pycutest_block{i}.txt')
                if os.path.exists(tout_file):
                    with open(tout_file, 'r') as f:
                        all_timeout.extend(f.read().split())
            
            with open(os.path.join(saving_path, 'feasibility_pycutest.txt'), 'w') as f:
                f.write(' '.join(sorted(list(set(all_feasibility)))))
            
            with open(os.path.join(saving_path, 'timeout_problems_pycutest.txt'), 'w') as f:
                f.write(' '.join(sorted(list(set(all_timeout)))))
            
            print("Merged feasibility and timeout info.")
        
    elif args_cmd.block is not None:
        block_num = args_cmd.block
        print(f"\n=== Processing block {block_num + 1} of {num_blocks} (Matrix Mode) ===")
        pycutest_clear_all_cache()
        collect_problem_info(block_num, problem_names_all)
    else:
        # Original sequential processing
        # Check which blocks are already finished
        files = os.listdir(saving_path)
        blocks_finished = []
        for f in files:
            if "probinfo_pycutest_block" in f and f.endswith(".csv"):
                match = re.findall(r"block(\d+)", f)
                if match:
                    blocks_finished.append(int(match[0]))
        blocks_finished.sort()

        # Process each block
        for block_num in range(num_blocks):
            if block_num in blocks_finished:
                print(f"\n=== Skipping block {block_num + 1} of {num_blocks} (already finished) ===\n")
                continue

            print(f"\n=== Processing block {block_num + 1} of {num_blocks} ===")

            # Clear all cache before each block to prevent memory buildup
            pycutest_clear_all_cache()

            collect_problem_info(block_num, problem_names_all)

            print(f"=== Finished block {block_num + 1} of {num_blocks} ===\n")

        # Automatically merge if running all blocks
        print("\n=== Merging all block files ===")
        dfs = []
        for i in range(num_blocks):
            block_file = os.path.join(saving_path, f'probinfo_pycutest_block{i}.csv')
            if os.path.exists(block_file):
                dfs.append(pd.read_csv(block_file))

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(os.path.join(saving_path, 'probinfo_pycutest.csv'), index=False)
            print(f"Merged {len(dfs)} block files into probinfo_pycutest.csv")

    print("\nScript completed successfully.")

    # Clean up logging
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
