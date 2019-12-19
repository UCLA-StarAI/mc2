import itertools
import os

DATASETS = {'mnist': 'data/mnist/mnist.pklz',
            'fmnist': 'data/fmnist/fmnist.pklz', }


TRIALS = 3
MISS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
TAYLOR = 5

PYTHON_INTERPRETER = '/opt/miniconda3/bin/ipython '
GRID_SCRIPT = 'run_missing_pair.py'
RAND_SEEDS = [1337, 123, 777, 9999]

BASE_EXP_DIR = 'exp'
EXP_ID = 'trial-III'
print('')
for d in DATASETS:
    for m in MISS:
        for t in range(TRIALS):
            CMD = ''
            CMD += f'{PYTHON_INTERPRETER} -- '
            CMD += f' {GRID_SCRIPT} '
            CMD += f' {DATASETS[d]} '

            out_dir = os.path.join(BASE_EXP_DIR, f'{d}-final')
            CMD += f' -o {out_dir}'

            v_tree_path = os.path.join(out_dir, f'{d}.vtree')
            CMD += f' --vtree {v_tree_path}'

            CMD += f' --seed {RAND_SEEDS[t]}'

            CMD += f' --miss-perc {m}'
            CMD += f' --taylor {TAYLOR}'
            CMD += f' --repeat 1'
            exp_id_str = f' {EXP_ID}-{m}-{t}'

            CMD += f' --exp-id {exp_id_str}'

            CMD += '\n'

            print(CMD)
