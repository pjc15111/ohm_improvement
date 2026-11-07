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
                          'NOISE_MODE', 'OI_SINGLE_EPOCHS', 'GPU_TIME', 'WPO_TIME', 'SEED_BASE', 'WPO_WORKERS'}}
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
    digital, hat, wpo, *ois, cleanup = [json.loads(line) for line in calls_file.read_text().splitlines()]
    assert len(ois) == 4
    bases = [args[args.index(str(folder / 'run.slurm')) + 3] for args in [digital, hat, wpo, *ois, cleanup]]
    assert len(set(bases)) == 1
    assert 0 <= int(bases[0]) < 2**32
    if seed_base is not None:
        assert bases[0] == seed_base
    assert '--dependency=afterok:1001' in hat
    assert '--dependency=aftercorr:1002' in wpo
    assert all('--dependency=afterok:1002' in args for args in ois)
    assert '--dependency=afterany:1001:1002:1003:1004:1005:1006:1007' in cleanup
    assert not any(arg.startswith('--dependency') for arg in digital)
    for args in [digital, hat, wpo, *ois]:
        assert '--array=1-5' in args
        assert '--kill-on-invalid-dep=yes' in args
        assert not any(arg.startswith('--constraint') for arg in args)
        assert '--time=02:00:00' in args
    for args in [digital, hat, *ois]:
        assert '--gres=gpu:1' in args and '--partition=test-gpu' in args
        assert '--mem=16G' in args
    assert '--partition=test-cpu' in wpo
    assert f'--cpus-per-task={wpo_workers or "64"}' in wpo
    assert f'--mem={wpo_mem or "64G"}' in wpo
    assert '--hint=nomultithread' in wpo
    assert digital[-1] == '14'
    assert hat[-1] == ('5' if hat_epochs is None else hat_epochs)
    assert [args[-4:] for args in ois] == [
        [epochs, count, initialization, noise_mode or 'fresh']
        for initialization in ['digital', 'hat'] for epochs, count in [('5', '12'), ('60', '1')]]
    assert not any(arg.startswith('--gres') for arg in wpo)

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
    commands = []
    for phase, arguments in [('digital', ['14']), ('hat', ['5']),
                              ('oi', ['5', '12', 'hat', 'fresh']),
                              ('oi', ['60', '1', 'hat', 'fresh']), ('wpo', [])]:
        completed = subprocess.run(['bash', str(MNIST / 'run.slurm'), phase,
                                    str(tmp_path / 'results'), '123456', *arguments],
                                   env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
        commands.append(json.loads(completed.stdout.splitlines()[-1]))
    for argv in commands:
        assert argv[argv.index('--seed') + 1] == '123458'
        assert argv[argv.index('--checkpoint') + 1].endswith('/checkpoints/seed-2.pt')
    shared_hat = commands[1][commands[1].index('--output-checkpoint') + 1]
    for argv in commands[2:]:
        assert argv[argv.index('--hat-checkpoint') + 1] == shared_hat
        assert argv[argv.index('--noise-seed') + 1] == '102'
    assert commands[-1][commands[-1].index('--workers') + 1] == '1'
    for argv, epochs, realizations, sampling in zip(commands[2:4], ['5', '60'], ['4', '1'], ['all', 'random']):
        assert argv[argv.index('--oi-epochs') + 1] == epochs
        assert argv[argv.index('--train-realizations') + 1] == realizations
        assert argv[argv.index('--time-sampling') + 1] == sampling
        assert argv[argv.index('--evaluation-realizations') + 1] == '4'
        assert argv[argv.index('--noise-mode') + 1] == 'fresh'
