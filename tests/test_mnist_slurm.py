"""Check SLURM orchestration without submitting real jobs."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

MNIST = Path(__file__).resolve().parents[1] / 'experiments' / 'mnist'


@pytest.mark.parametrize('hat_epochs', [None, '0', '2'])
@pytest.mark.parametrize('wpo_mem,wpo_workers', [(None, None), ('8G', '1')])
@pytest.mark.parametrize('noise_mode', [None, 'fixed'])
@pytest.mark.parametrize('seed_base', [None, '123456'])
def test_submission_dependencies_resources_and_checkpoint_cleanup(tmp_path, hat_epochs, wpo_mem, wpo_workers, noise_mode, seed_base):
    # A copied checkout keeps generated result directories inside this test.
    folder = tmp_path / 'experiments' / 'mnist'
    folder.mkdir(parents=True)
    for name in ['submit.sh', 'run.slurm']:
        shutil.copyfile(MNIST / name, folder / name)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    calls_file = tmp_path / 'calls.jsonl'
    sbatch = bin_dir / 'sbatch'
    sbatch.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
path = Path(os.environ['SBATCH_CALLS'])
with path.open('a') as file:
    file.write(json.dumps(sys.argv[1:]) + '\\n')
print(1000 + len(path.read_text().splitlines()))
''')
    sbatch.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in {'RUNS', 'DIGITAL_EPOCHS', 'OI_EPOCHS', 'HAT_EPOCHS', 'WPO_MEM',
                          'NOISE_MODE', 'OI_LONG_EPOCHS', 'GPU_TIME', 'WPO_TIME', 'SEED_BASE', 'WPO_WORKERS'}}
    if wpo_workers is not None:
        env['WPO_WORKERS'] = wpo_workers
    if seed_base is not None:
        env['SEED_BASE'] = seed_base
    if hat_epochs is not None:
        env['HAT_EPOCHS'] = hat_epochs
    if wpo_mem is not None:
        env['WPO_MEM'] = wpo_mem
    if noise_mode is not None:
        env['NOISE_MODE'] = noise_mode
    env.update({'PATH': f'{bin_dir}:{os.environ["PATH"]}',
           'SBATCH_CALLS': str(calls_file),
           'GPU_PARTITION': 'test-gpu', 'CPU_PARTITION': 'test-cpu'})
    subprocess.run(['bash', str(folder / 'submit.sh')], env=env, check=True,
                   capture_output=True, text=True)
    digital, hat, wpo_digital, oi_d10, oi_d20, wpo_hat, oi_h10, oi_h20, cleanup = [json.loads(line) for line in calls_file.read_text().splitlines()]
    wpos = [wpo_digital, wpo_hat]
    ois = [oi_d10, oi_d20, oi_h10, oi_h20]
    assert len(ois) == 4
    bases = [args[args.index(str(folder / 'run.slurm')) + 3] for args in [digital, hat, *wpos, *ois, cleanup]]
    assert len(set(bases)) == 1
    assert 0 <= int(bases[0]) < 2**32
    if seed_base is not None:
        assert bases[0] == seed_base
    assert '--dependency=afterok:1001' in hat
    assert '--dependency=aftercorr:1001' in wpo_digital
    assert '--dependency=aftercorr:1002' in wpo_hat
    assert all('--dependency=afterok:1002' in args for args in ois)
    assert '--dependency=afterany:1001:1002:1003:1004:1005:1006:1007:1008' in cleanup
    assert not any(arg.startswith('--dependency') for arg in digital)
    for args in [digital, hat, *wpos, *ois]:
        assert '--array=1-5' in args
        assert '--kill-on-invalid-dep=yes' in args
        assert not any(arg.startswith('--constraint') for arg in args)
        assert '--time=02:00:00' in args
    for args in [digital, hat, *ois]:
        assert '--gres=gpu:1' in args and '--partition=test-gpu' in args
        assert '--mem=16G' in args
    assert all('--partition=test-cpu' in job for job in wpos)
    assert all(f'--cpus-per-task={wpo_workers or "64"}' in job for job in wpos)
    assert all(f'--mem={wpo_mem or "64G"}' in job for job in wpos)
    assert all('--hint=nomultithread' in job for job in wpos)
    assert digital[-1] == '10'
    assert hat[-1] == ('10' if hat_epochs is None else hat_epochs)
    assert [args[-3:] for args in ois] == [
        [epochs, initialization, noise_mode or 'fresh']
        for initialization in ['digital', 'hat'] for epochs in ['10', '20']]
    assert [args[-1] for args in wpos] == ['digital', 'hat']
    assert all(not any(arg.startswith('--gres') for arg in job) for job in wpos)

    run_dir = next((tmp_path / 'results' / 'mnist').glob('run-*'))
    checkpoint = run_dir / 'checkpoints' / 'seed-1.pt'
    checkpoint.touch()
    (run_dir / 'checkpoints' / 'hat-seed-1.pt').touch()
    history = run_dir / 'seed-1' / 'digital.json'
    history.write_text('{}\n')
    subprocess.run(['bash', str(MNIST / 'run.slurm'), 'cleanup', str(run_dir), bases[0]],
                   cwd=MNIST.parents[1], check=True, capture_output=True, text=True)
    assert not (run_dir / 'checkpoints').exists()
    assert history.read_text() == '{}\n'


def test_runner_uses_fresh_training_seed_and_one_shared_hat_checkpoint(tmp_path):
    python_bin = tmp_path / 'venv' / 'bin' / 'python'
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text('#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    python_bin.chmod(0o755)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    git = bin_dir / 'git'
    git.write_text('#!/bin/sh\nprintf "test-commit\\n"\n')
    git.chmod(0o755)
    env = {**os.environ, 'SLURM_ARRAY_TASK_ID': '2', 'PATH': f'{bin_dir}:{os.environ["PATH"]}',
           'WPO_WORKERS': '1'}
    conditions = [('digital', ['10']), ('hat', ['10']),
                  ('oi', ['10', 'hat', 'fresh']), ('oi', ['20', 'hat', 'fresh']),
                  ('oi', ['10', 'digital', 'fresh']), ('oi', ['20', 'digital', 'fresh']),
                  ('wpo', ['hat']), ('wpo', ['digital'])]
    commands = []
    for phase, arguments in conditions:
        completed = subprocess.run(['bash', str(MNIST / 'run.slurm'), phase,
                                    str(tmp_path / 'results'), '123456', *arguments],
                                   env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
        commands.append(json.loads(completed.stdout.splitlines()[-1]))
    shared_hat = commands[1][commands[1].index('--output-checkpoint') + 1]
    for (phase, arguments), argv in zip(conditions, commands):
        assert argv[argv.index('--seed') + 1] == '123458'
        assert argv[argv.index('--checkpoint') + 1].endswith('/checkpoints/seed-2.pt')
        if phase in ['oi', 'wpo']:
            assert argv[argv.index('--noise-seed') + 1] == '102'
            initialization = arguments[1] if phase == 'oi' else arguments[0]
            if initialization == 'hat':
                assert argv[argv.index('--hat-checkpoint') + 1] == shared_hat
            else:
                assert '--hat-checkpoint' not in argv
        if phase == 'oi':
            assert argv[argv.index('--oi-epochs') + 1] == arguments[0]
            for flag, value in [('--train-realizations', '1'), ('--time-sampling', 'random'),
                                ('--conductance-lr', '0.001'), ('--significance-lr', '0.00001'),
                                ('--bias-lr', '0.00001'), ('--output-scaling-lr', '0.00001'),
                                ('--evaluation-realizations', '4'), ('--noise-mode', 'fresh')]:
                assert argv[argv.index(flag) + 1] == value
            assert '--optimize-bias' in argv and '--optimize-output-scaling' in argv
            assert argv[argv.index('--results-file') + 1].endswith(f'/oi-{arguments[1]}-e{arguments[0]}.json')
        if phase == 'wpo':
            assert argv[argv.index('--workers') + 1] == '1'
            assert argv[argv.index('--results-file') + 1].endswith(f'/wpo-{arguments[0]}.json')
