import itertools
import os

DATASETS = [
    'data/boston/boston.pklz',
    'data/appliances/appliances.pklz',
    'data/abalone/abalone.pklz',
    'data/insurance/insurance.pklz',
    'data/kings-county/kings-county.pklz',
    'data/delta-ailerons/delta-ailerons.pklz',
    'data/elevators/elevators.pklz',
    'data/kinematics/kinematics.pklz',
    'data/compact/compact.pklz'
]
ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100]
DEPTHS = [2, 10, 20]
SPLITS = [1, 3]
# SL_ITERS = [10, 100]
# PL_ITERS = [10, 100]
SL_ITERS = [100]
PL_ITERS = [100]

PYTHON_INTERPRETER = '/opt/miniconda3/bin/ipython '
GRID_SCRIPT = 'reg_circuit_grid.py'
EXP_DIR = 'exp/new-reg-circuit-grid'
PATIENCE = 5
VALIDATE = 5
RAND_SEED = 1337

print('')
for d in DATASETS:
    CMD = ''
    CMD += f'{PYTHON_INTERPRETER} -- '
    CMD += f' {GRID_SCRIPT} '
    CMD += f' {d} '

    d_name = os.path.basename(d).replace('.pklz', '')
    out_dir = os.path.join(EXP_DIR, d_name)
    CMD += f' -o {out_dir}'

    CMD += f' --seed {RAND_SEED}'

    alpha_str = ' '.join(str(a) for a in ALPHAS)
    CMD += f' --alpha {alpha_str}'
    sl_str = ' '.join(str(a) for a in SL_ITERS)
    CMD += f' --n-iter-sl {sl_str}'
    pl_str = ' '.join(str(a) for a in PL_ITERS)
    CMD += f' --n-iter-pl {pl_str}'
    depth_str = ' '.join(str(a) for a in DEPTHS)
    CMD += f' --depth {depth_str}'
    split_str = ' '.join(str(a) for a in SPLITS)
    CMD += f' --n-splits {split_str}'
    CMD += f' --patience {PATIENCE}'
    CMD += f' --validate-every {VALIDATE}'
    CMD += ' --vtree "balanced"'
    CMD += '\n'

    print(CMD)
