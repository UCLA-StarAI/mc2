import itertools
import os

DATASETS = {
    'boston':   'data/boston/boston.pklz',
    'appliances': 'data/appliances/appliances.pklz',
    'abalone': 'data/abalone/abalone.pklz',
    'insurance': 'data/insurance/insurance.pklz',
    'kings-county': 'data/kings-county/kings-county.pklz',
    'delta-ailerons': 'data/delta-ailerons/delta-ailerons.pklz',
    'elevators': 'data/elevators/elevators.pklz',
    'kinematics': 'data/kinematics/kinematics.pklz',
    'compact': 'data/compact/compact.pklz'
}

DATASETS_EXP_PATH = {
    'boston':   'boston_20190520-180922/',
    'appliances': 'appliances_20190520-180922',
    'abalone': 'abalone_20190520-182501',
    'insurance': 'insurance_20190520-184809',
    'kings-county': 'kings-county_20190520-193104',
    'delta-ailerons': 'delta-ailerons_20190520-190320',
    'elevators': 'elevators_20190520-185859',
    'kinematics': 'kinematics_20190520-234527',
    'compact': 'compact_20190521-002416'
}

EXP_BASE_PATH = 'exp/new-reg-circuit-grid/'

PYTHON_INTERPRETER = '/opt/miniconda3/bin/ipython '
MISS_SCRIPT = 'run_missing_pair.py'
PLOT_SCRIPT = 'plot_missing_pair.py'
MISS_PERC = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
REPEAT = 10
EXP_ID = ' try-I'

print('')
for d in DATASETS:

    data_path = DATASETS[d]

    CMD = ''
    # CMD += f'{PYTHON_INTERPRETER} -- '
    # CMD += f' {MISS_SCRIPT} '
    # CMD += f' {data_path} '

    # out_dir = os.path.join(EXP_BASE_PATH, d, DATASETS_EXP_PATH[d], 'best')
    # CMD += f' -o {out_dir}'
    # CMD += ' --miss-perc {}'.format(' '.join(str(k) for k in MISS_PERC))
    # CMD += f' --repeat {REPEAT}'
    # CMD += ' --regression'
    # CMD += f' --exp-id {EXP_ID}'

    # CMD += f' &> ./{d}.log  && '

    #
    # plotting
    CMD += f' {PYTHON_INTERPRETER} -- '
    CMD += f' {PLOT_SCRIPT} '
    CMD += f' {data_path} '

    out_dir = os.path.join(EXP_BASE_PATH, d, DATASETS_EXP_PATH[d], 'best')
    CMD += f' -o {out_dir}'

    v_tree_path = os.path.join(EXP_BASE_PATH, d, DATASETS_EXP_PATH[d], f'{d}.vtree')
    CMD += f' --vtree {v_tree_path}'

    CMD += ' --regression'
    CMD += f' --exp-id {EXP_ID}'
    CMD += ' --do-mpe'
    CMD += ' --do-mice'
    CMD += f' &> ./{d}.log'

    CMD += '\n'

    print(CMD)
